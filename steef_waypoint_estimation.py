#!/usr/bin/env python3
# plane_ransac_halfspace.py

import sys
import os
import time
import argparse
import numpy as np
import open3d as o3d
from centerline import compute_centerline, visualize_with_centerline
from pcd_clean_radius import estimate_radius, keep_largest_component_radius
# ---------------------------
# Helpers
# ---------------------------

def load_and_downsample(path: str, voxel_size: float = 0.5) -> o3d.geometry.PointCloud:
    # time this function
    start_time = time.time()
    pcd = o3d.io.read_point_cloud(path)
    end_time = time.time()
    print(f"Loaded point cloud in {end_time - start_time:.2f} seconds")
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {path}")
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def segment_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
):
    """
    Returns (plane_model, inlier_indices).
    plane_model is (a, b, c, d) where ax + by + cz + d = 0
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return plane_model, inliers


def normalize_plane(plane_model):
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n) + 1e-12
    return np.r_[n / n_norm, d / n_norm]  # (a', b', c', d')


def split_by_halfspace(
    pcd: o3d.geometry.PointCloud,
    plane_model,
    keep_above=True,
    margin=0.0
):
    """
    keep_above=True  -> keep s >= -margin   (on/above plane along +normal)
    keep_above=False -> keep s <  -margin   (strictly below plane)
    """
    a, b, c, d = normalize_plane(plane_model)
    pts = np.asarray(pcd.points)
    s = pts @ np.array([a, b, c]) + d  # signed distance
    if keep_above:
        idx = np.where(s >= -margin)[0]
    else:
        idx = np.where(s < -margin)[0]
    kept = pcd.select_by_index(idx)
    dropped = pcd.select_by_index(idx, invert=True)
    return kept, dropped, s


def colorize_and_split(
    pcd: o3d.geometry.PointCloud,
    inliers: list,
    plane_color=(1.0, 0.0, 0.0),
    others_color=(0.6, 0.6, 0.6),
):
    plane_cloud = pcd.select_by_index(inliers)
    rest_cloud = pcd.select_by_index(inliers, invert=True)
    plane_cloud.paint_uniform_color(plane_color)
    rest_cloud.paint_uniform_color(others_color)
    return plane_cloud, rest_cloud


def make_plane_patch_from_inliers(
    plane_cloud: o3d.geometry.PointCloud,
    color=(0.0, 0.4, 1.0),
    pad=0.0
) -> o3d.geometry.TriangleMesh:
    """
    Create a thin rectangular mesh lying on the plane, sized to the inlier footprint.
    Uses OBB of inliers to get 2D extents; the smallest extent is treated as the normal axis.
    """
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(plane_cloud.points)
    ex, ey, ez = obb.extent
    R = obb.R
    c = obb.center

    # Which axis is (near) normal? choose the smallest extent
    extents = np.array([ex, ey, ez], dtype=float)
    normal_axis = int(np.argmin(extents))
    in_plane_axes = [i for i in range(3) if i != normal_axis]
    e1, e2 = extents[in_plane_axes] * 0.5 + pad
    u = R[:, in_plane_axes[0]]
    v = R[:, in_plane_axes[1]]

    corners = np.array([
        c + (+e1) * u + (+e2) * v,
        c + (-e1) * u + (+e2) * v,
        c + (-e1) * u + (-e2) * v,
        c + (+e1) * u + (-e2) * v,
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def keep_largest_cluster(pcd: o3d.geometry.PointCloud,
                         eps: float = 3.0,
                         min_points: int = 100) -> o3d.geometry.PointCloud:
    """
    Cluster points with DBSCAN and keep only the largest cluster (label with most points).
    eps: neighborhood radius (same units as point cloud)
    min_points: minimum points to form a core cluster
    """
    if len(pcd.points) == 0:
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        # No clusters found (all noise) -> return as-is
        return pcd

    # Count only non-noise labels (>= 0)
    valid = labels >= 0
    if not np.any(valid):
        return pcd

    largest_label = np.bincount(labels[valid]).argmax()
    idx = np.where(labels == largest_label)[0]
    kept = pcd.select_by_index(idx)
    print(f"[cluster] clusters={labels.max()+1}, kept label={largest_label} "
          f"({len(idx)} pts), dropped {len(labels) - len(idx)} pts as other clusters/noise")
    return kept

def save_outputs(
    out_dir: str,
    plane_cloud: o3d.geometry.PointCloud,
    rest_cloud: o3d.geometry.PointCloud,
    kept_above: o3d.geometry.PointCloud | None = None
):
    os.makedirs(out_dir, exist_ok=True)
    combo = plane_cloud + rest_cloud
    o3d.io.write_point_cloud(os.path.join(out_dir, "colored_combined.ply"), combo)
    o3d.io.write_point_cloud(os.path.join(out_dir, "plane_inliers.ply"), plane_cloud)
    o3d.io.write_point_cloud(os.path.join(out_dir, "others_outliers.ply"), rest_cloud)
    if kept_above is not None:
        o3d.io.write_point_cloud(os.path.join(out_dir, "kept_above_plane.ply"), kept_above)


# ---------------------------
# Main
# ---------------------------

def main():
    #start timer

    parser = argparse.ArgumentParser(description="RANSAC plane segmentation + half-space filtering")
    parser.add_argument("--pcd", type=str, default="./data/08.pcd", help="Input point cloud")
    parser.add_argument("--voxel", type=float, default=0.5, help="Voxel size for downsampling (0 to disable)")
    parser.add_argument("--dist", type=float, default=0.5, help="RANSAC distance threshold")
    parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC sample size")
    parser.add_argument("--iters", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument("--plane_color", type=str, default="1,0,0", help="RGB for plane (0-1, comma sep)")
    parser.add_argument("--others_color", type=str, default="0.6,0.6,0.6", help="RGB for others (0-1, comma sep)")
    parser.add_argument("--below_margin", type=float, default=0.0, help="Tolerance band below plane")
    parser.add_argument("--keep_above", action="store_true", help="Drop all points below plane normal")
    parser.add_argument("--force_up_normal", action="store_true",
                        help="Flip plane normal to align with +Z before filtering")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save PLYs")
    parser.add_argument("--cluster_eps", type=float, default=3.0,
                    help="DBSCAN eps (neighborhood radius)")
    parser.add_argument("--cluster_min_points", type=int, default=30,
                    help="DBSCAN min points per cluster")
    args = parser.parse_args()

    # Parse colors
    plane_rgb = tuple(map(float, args.plane_color.split(",")))
    others_rgb = tuple(map(float, args.others_color.split(",")))
    # Load & downsample
    try:
        originalpcd = load_and_downsample(args.pcd, args.voxel)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
        
    start_time = time.time()
    pcd = originalpcd
    # Segment dominant plane
    try:
        plane_model, inliers = segment_plane(
            pcd, distance_threshold=args.dist, ransac_n=args.ransac_n, num_iterations=args.iters
        )
    except Exception as e:
        print(f"RANSAC failed: {e}")
        sys.exit(1)

    if len(inliers) == 0:
        print("No plane detected with the given parameters.")
        sys.exit(2)

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    if args.force_up_normal and normal @ np.array([0.0, 0.0, 1.0]) < 0:
        plane_model = (-a, -b, -c, -d)
        normal = -normal

    # Colorized plane/non-plane clouds
    plane_cloud, rest_cloud = colorize_and_split(pcd, inliers, plane_rgb, others_rgb)

    # Visible plane patch mesh sized to inlier footprint
    plane_patch = make_plane_patch_from_inliers(plane_cloud, color=(0.0, 0.4, 1.0), pad=0.0)
    
    # Post plane filtering
    pcd = pcd.voxel_down_sample(voxel_size=1)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Only keep biggest blob
    pcd = keep_largest_cluster(pcd, eps=args.cluster_eps, min_points=args.cluster_min_points)

    # Optional: remove points "below" the plane (w.r.t. plane normal)
    kept_above = None
    if args.keep_above:
        kept_above, dropped, s = split_by_halfspace(pcd, plane_model, keep_above=True, margin=args.below_margin)
        kept_above.paint_uniform_color([1, 0, 0])
        dropped.paint_uniform_color([0.7, 0.7, 0.7])
        plane_cloud.paint_uniform_color(plane_rgb)
        print(f"Kept {len(kept_above.points)} on/above plane; removed {len(dropped.points)} below "
              f"(margin={args.below_margin}).")
        """o3d.visualization.draw_geometries(
            [kept_above,dropped],
            window_name="Only above-plane points"
        )"""
    else:
        o3d.visualization.draw_geometries([rest_cloud, plane_cloud, plane_patch],
                                          window_name="Plane (colored) vs Others + plane patch")

    # Save outputs
    save_outputs(args.save_dir, plane_cloud, rest_cloud, kept_above)

    # Print plane details
    n_unit = normal / (np.linalg.norm(normal) + 1e-12)
    print(f"Plane: {a:.6f} x + {b:.6f} y + {c:.6f} z + {d:.6f} = 0   | normal = {n_unit}")
    print(f"Execution Time: {time.time() - start_time:.2f} s")

    kept_above = keep_largest_component_radius(kept_above, 2, min_component_size=0)

    # 2) Compute the centerline
    center_pts, center_ls = compute_centerline(
        kept_above,
        n_slices=500,      # increase if the cloud is long and detailed
        min_points=2,    # min points per slice to keep a centroid
        smooth_window=7,   # odd integer; set 0/1 to disable smoothing
        force_endpoints=True
    )
    # Stop timer
    end_time = time.time()
    print(f"Centerline computation time: {end_time - start_time:.2f} s")
    # 3) Visualize (point cloud + centerline + waypoint spheres)
    visualize_with_centerline(kept_above, center_pts)

if __name__ == "__main__":
    main()
