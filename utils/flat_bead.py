import open3d as o3d
import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import entropy
import matplotlib.pyplot as plt
from utils.centerline2graph import clean_and_smooth_centerline

def create_coordinate_frame(size=10.0):
    """Creates an Open3D LineSet object representing a coordinate frame."""
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector([[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
    frame.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    frame.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # X=Red, Y=Green, Z=Blue
    return frame

def create_xy_plane(size=50.0):
    """Creates a large, semi-transparent plane mesh on the XY plane at Z=0."""
    plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.01)
    plane.translate([-size/2, -size/2, -0.01])
    plane.paint_uniform_color([0.7, 0.7, 0.7])
    plane.compute_vertex_normals()
    return plane

def align_pcd_to_principal_axes(pcd):
    """Aligns the point cloud so its longest dimension is along the Y-axis."""
    print("Aligning point cloud to its principal axes...")
    mean = pcd.get_center()
    pcd.translate(-mean)
    cov_matrix = np.cov(np.asarray(pcd.points), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sort_order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sort_order]
    principal_axis_short = eigenvectors[:, 0]
    principal_axis_medium = eigenvectors[:, 1]
    principal_axis_long = eigenvectors[:, 2]
    target_z = np.array([0.0, 0.0, 1.0])
    if np.dot(principal_axis_short, target_z) < 0:
        principal_axis_short *= -1
    rotation_matrix = np.stack([principal_axis_medium, principal_axis_long, principal_axis_short]).T
    pcd.rotate(np.linalg.inv(rotation_matrix), center=(0,0,0))
    print("PCA alignment complete.")
    return pcd

def contract_to_centerline(bead_pcd, num_iterations=1000, contraction_factor=0.01):
    """Contracts a point cloud to its centerline/skeleton using Laplacian smoothing."""
    print("Contracting points to find the centerline/skeleton...")
    if not bead_pcd.has_points(): return o3d.geometry.PointCloud()
    points = np.asarray(bead_pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(bead_pcd)
    num_points = len(points)
    adj_matrix_rows, adj_matrix_cols = [], []
    for i in range(num_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], knn=10)
        adj_matrix_rows.extend([i] * (k - 1))
        adj_matrix_cols.extend(idx[1:])
    adj_data = np.ones(len(adj_matrix_rows))
    A = csc_matrix((adj_data, (adj_matrix_rows, adj_matrix_cols)), shape=(num_points, num_points))
    A = (A + A.T) / 2
    D = csc_matrix((A.sum(axis=1).A1, (range(num_points), range(num_points))))
    L = D - A
    contracted_points = points.copy()
    for _ in range(num_iterations):
        delta = L.dot(contracted_points)
        contracted_points -= contraction_factor * delta
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(contracted_points)
    return centerline_pcd

def segment_bead_with_ransac(aligned_pcd, distance_threshold=0.1):
    """Segments the bead from the plane using RANSAC plane fitting."""
    print(f"Segmenting plane with RANSAC (distance threshold: {distance_threshold})...")
    plane_model, inliers = aligned_pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    plane_pcd = aligned_pcd.select_by_index(inliers)
    bead_pcd = aligned_pcd.select_by_index(inliers, invert=True)
    print(f"RANSAC isolated {len(bead_pcd.points)} points for the bead.")
    return bead_pcd, plane_pcd

def segment_bead_with_zscore(aligned_pcd, z_score_threshold=1.5):
    """Segments the bead using Z-score filtering on the Z-axis height."""
    print(f"Segmenting bead using Z-score filtering (threshold: {z_score_threshold})...")
    points = np.asarray(aligned_pcd.points)
    z_coords = points[:, 2]
    mean_z = np.mean(z_coords)
    std_z = np.std(z_coords)
    z_scores = (z_coords - mean_z) / std_z
    bead_indices = np.where(z_scores > z_score_threshold)[0]
    plane_indices = np.where(z_scores <= z_score_threshold)[0]
    plane_pcd = aligned_pcd.select_by_index(plane_indices)
    bead_pcd = aligned_pcd.select_by_index(bead_indices)
    print(f"Z-score isolated {len(bead_pcd.points)} points for the bead.")
    return bead_pcd, plane_pcd

def segment_bead_by_surface_subtraction(aligned_pcd, difference_threshold=0.1):
    """Segments the bead by subtracting a fitted polynomial surface from the original cloud."""
    print("Segmenting bead by subtracting a fitted surface...")
    
    _, initial_plane_pcd = segment_bead_with_zscore(aligned_pcd, z_score_threshold=1.0)
    plane_points = np.asarray(initial_plane_pcd.points)
    
    if len(plane_points) < 10:
        print("Warning: Not enough plane points found to fit a surface.")
        return o3d.geometry.PointCloud(), aligned_pcd

    x = plane_points[:, 0]
    y = plane_points[:, 1]
    z = plane_points[:, 2]
    A = np.c_[np.ones(x.shape[0]), x, y, x*y, x**2, y**2]
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    
    all_points = np.asarray(aligned_pcd.points)
    all_x = all_points[:, 0]
    all_y = all_points[:, 1]
    all_z_actual = all_points[:, 2]
    
    all_A = np.c_[np.ones(all_x.shape[0]), all_x, all_y, all_x*all_y, all_x**2, all_y**2]
    z_predicted = np.dot(all_A, coeffs)
    
    difference = all_z_actual - z_predicted
    bead_indices = np.where(difference > difference_threshold)[0]
    plane_indices = np.where(difference <= difference_threshold)[0]
    
    plane_pcd = aligned_pcd.select_by_index(plane_indices)
    bead_pcd = aligned_pcd.select_by_index(bead_indices)
    
    print(f"Surface subtraction isolated {len(bead_pcd.points)} points for the bead.")
    return bead_pcd, plane_pcd


def run(input_pcd):
    """Main function to run the bead segmentation and centerline extraction."""
    print("Starting bead segmentation and centerline extraction...")

    # Step 1: Align the point cloud to its principal axes
    aligned_pcd = align_pcd_to_principal_axes(input_pcd)
    
    coord_frame = create_coordinate_frame(size=20.0)
    # xy_plane = create_xy_plane(size=100.0)
    print("\n--- PCA Alignment Visualization ---")
    o3d.visualization.draw_geometries([aligned_pcd, coord_frame], window_name="PCA Aligned Point Cloud")

    # --- Method 3: Surface Subtraction Segmentation ---
    bead_pcd_surf, plane_pcd_surf = segment_bead_by_surface_subtraction(aligned_pcd, difference_threshold=0.75)
    
    # Visualize the blobs before filtering to help with parameter tuning.
    print("Visualizing detected blobs for parameter tuning...")
    if bead_pcd_surf.has_points():
        eps_for_blobs = 1.0
        min_points_for_blobs = 20
        labels = np.array(bead_pcd_surf.cluster_dbscan(eps=eps_for_blobs, min_points=min_points_for_blobs, print_progress=False))
        
        max_label = labels.max()
        print(f"Found {max_label + 1} blobs with eps={eps_for_blobs} and min_points={min_points_for_blobs}.")
        
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        
        blobs_pcd = o3d.geometry.PointCloud()
        blobs_pcd.points = bead_pcd_surf.points
        blobs_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        o3d.visualization.draw_geometries([blobs_pcd], window_name="Blob Visualization for Tuning")

    # **NEW**: Filter blobs to keep ALL large blobs, not just the single largest one.
    print("Filtering small blobs from surface subtraction result...")
    if bead_pcd_surf.has_points():
        labels = np.array(bead_pcd_surf.cluster_dbscan(eps=1.0, min_points=20, print_progress=False))
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        non_noise_mask = unique_labels != -1
        unique_labels = unique_labels[non_noise_mask]
        counts = counts[non_noise_mask]
        
        if len(counts) > 0:
            # Find the size of the largest blob
            largest_blob_size = counts.max()
            # Keep all blobs that are at least 25% of the size of the largest one
            size_threshold = largest_blob_size * 0.25
            
            # Get the labels of all blobs that meet the threshold
            large_enough_labels = unique_labels[counts >= size_threshold]
            
            # Create a boolean mask to select points belonging to any of the large blobs
            final_bead_mask = np.isin(labels, large_enough_labels)
            final_bead_indices = np.where(final_bead_mask)[0]
            
            bead_pcd_surf = bead_pcd_surf.select_by_index(final_bead_indices)

    plane_pcd_surf.paint_uniform_color([0.8, 0.8, 0.8])
    bead_pcd_surf.paint_uniform_color([0, 1, 1]) # Cyan
    print("Visualizing Surface Subtraction segmentation (after blob filtering)...")
    o3d.visualization.draw_geometries([plane_pcd_surf, bead_pcd_surf], window_name="3. Surface Subtraction Segmentation (Cleaned)")


    # --- Proceeding with Surface Subtraction result for centerline extraction ---
    print("\n--- Centerline Extraction (using Surface Subtraction result) ---")
    if not bead_pcd_surf.has_points():
        print("No points found for the bead after segmentation. Exiting.")
        exit()
        
    cleaned_bead_pcd, _ = bead_pcd_surf.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    cleaned_bead_pcd.paint_uniform_color([0.9, 0.5, 0.5])

    centerline_pcd = contract_to_centerline(cleaned_bead_pcd, num_iterations=10000, contraction_factor=0.01)
    centerline_pcd.paint_uniform_color([0, 0, 1]) # Blue
    
    graph_centerline = clean_and_smooth_centerline(centerline_pcd, num_waypoints=200, k_neighbors=10, voxel_size=0.5)
    graph_centerline.paint_uniform_color([0, 1, 0]) # Blue

    print("\n--- Final Result Visualization ---")
    o3d.visualization.draw_geometries(
        [plane_pcd_surf, cleaned_bead_pcd, centerline_pcd, graph_centerline, coord_frame],
        window_name="Final Result with Full Context",
        width=1000, height=800
    )


if __name__ == "__main__":
    try:
        pcd = o3d.io.read_point_cloud("../data/seam_2/seam_2_seg_2.pcd")
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    print("\n--- Initial Visualization ---")
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    aligned_pcd = align_pcd_to_principal_axes(pcd)
    coord_frame = create_coordinate_frame(size=20.0)
    # xy_plane = create_xy_plane(size=100.0)
    print("\n--- PCA Alignment Visualization ---")
    o3d.visualization.draw_geometries([aligned_pcd, coord_frame], window_name="PCA Aligned Point Cloud")

    # --- Method 3: Surface Subtraction Segmentation ---
    bead_pcd_surf, plane_pcd_surf = segment_bead_by_surface_subtraction(aligned_pcd, difference_threshold=0.75)
    
    # Visualize the blobs before filtering to help with parameter tuning.
    print("Visualizing detected blobs for parameter tuning...")
    if bead_pcd_surf.has_points():
        eps_for_blobs = 1.0
        min_points_for_blobs = 20
        labels = np.array(bead_pcd_surf.cluster_dbscan(eps=eps_for_blobs, min_points=min_points_for_blobs, print_progress=False))
        
        max_label = labels.max()
        print(f"Found {max_label + 1} blobs with eps={eps_for_blobs} and min_points={min_points_for_blobs}.")
        
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        
        blobs_pcd = o3d.geometry.PointCloud()
        blobs_pcd.points = bead_pcd_surf.points
        blobs_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        o3d.visualization.draw_geometries([blobs_pcd], window_name="Blob Visualization for Tuning")

    # **NEW**: Filter blobs to keep ALL large blobs, not just the single largest one.
    print("Filtering small blobs from surface subtraction result...")
    if bead_pcd_surf.has_points():
        labels = np.array(bead_pcd_surf.cluster_dbscan(eps=1.0, min_points=20, print_progress=False))
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        non_noise_mask = unique_labels != -1
        unique_labels = unique_labels[non_noise_mask]
        counts = counts[non_noise_mask]
        
        if len(counts) > 0:
            # Find the size of the largest blob
            largest_blob_size = counts.max()
            # Keep all blobs that are at least 25% of the size of the largest one
            size_threshold = largest_blob_size * 0.25
            
            # Get the labels of all blobs that meet the threshold
            large_enough_labels = unique_labels[counts >= size_threshold]
            
            # Create a boolean mask to select points belonging to any of the large blobs
            final_bead_mask = np.isin(labels, large_enough_labels)
            final_bead_indices = np.where(final_bead_mask)[0]
            
            bead_pcd_surf = bead_pcd_surf.select_by_index(final_bead_indices)

    plane_pcd_surf.paint_uniform_color([0.8, 0.8, 0.8])
    bead_pcd_surf.paint_uniform_color([0, 1, 1]) # Cyan
    print("Visualizing Surface Subtraction segmentation (after blob filtering)...")
    o3d.visualization.draw_geometries([plane_pcd_surf, bead_pcd_surf], window_name="3. Surface Subtraction Segmentation (Cleaned)")


    # --- Proceeding with Surface Subtraction result for centerline extraction ---
    print("\n--- Centerline Extraction (using Surface Subtraction result) ---")
    if not bead_pcd_surf.has_points():
        print("No points found for the bead after segmentation. Exiting.")
        exit()
        
    cleaned_bead_pcd, _ = bead_pcd_surf.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    cleaned_bead_pcd.paint_uniform_color([0.9, 0.5, 0.5])

    centerline_pcd = contract_to_centerline(cleaned_bead_pcd, num_iterations=10000, contraction_factor=0.01)
    centerline_pcd.paint_uniform_color([0, 0, 1]) # Blue
    
    graph_centerline = clean_and_smooth_centerline(centerline_pcd, num_waypoints=200, k_neighbors=10, voxel_size=0.5)
    graph_centerline.paint_uniform_color([0, 1, 0]) # Blue

    print("\n--- Final Result Visualization ---")
    o3d.visualization.draw_geometries(
        [plane_pcd_surf, cleaned_bead_pcd, centerline_pcd, graph_centerline, coord_frame],
        window_name="Final Result with Full Context",
        width=1000, height=800
    )
