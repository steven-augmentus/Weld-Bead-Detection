import open3d as o3d
import numpy as np
from scipy.sparse import csc_matrix
from utils.centerline2graph import clean_and_smooth_centerline

def segment_corner_weld(pcd, eps=0.1, min_points=100, normal_consistency_angle_deg=10.0):
    """
    Segments a corner weld by finding the two largest planes via normal clustering,
    refines the result by filtering based on normal consistency, and finally
    cleans the resulting bead by keeping only the largest connected component.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        eps (float): DBSCAN epsilon parameter for clustering normals.
        min_points (int): DBSCAN min_points parameter.
        normal_consistency_angle_deg (float): Max angle (in degrees) a point's normal
                                              can deviate from the plane's average normal.

    Returns:
        tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
            A tuple containing the final cleaned bead, plane 1, and plane 2 point clouds.
    """
    print("Estimating normals for segmentation...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=100)

    if not pcd.has_normals():
        print("Error: Normal estimation failed.")
        return None, None, None

    normals = np.asarray(pcd.normals)

    # --- Step 1: Cluster the normal vectors using DBSCAN ---
    print("Clustering normal vectors to find dominant planes...")
    normals_pcd = o3d.geometry.PointCloud()
    normals_pcd.points = o3d.utility.Vector3dVector(normals)
    
    labels = np.array(normals_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    max_label = labels.max()
    print(f"Found {max_label + 1} distinct plane clusters.")

    # --- Step 2: Identify the two largest clusters ---
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_noise_mask = unique_labels != -1
    unique_labels = unique_labels[non_noise_mask]
    counts = counts[non_noise_mask]

    if len(counts) < 2:
        print("Error: Could not find at least two distinct planes. Try adjusting DBSCAN parameters.")
        return None, None, None

    sorted_count_indices = np.argsort(-counts)
    plane1_label = unique_labels[sorted_count_indices[0]]
    plane2_label = unique_labels[sorted_count_indices[1]]

    # --- Step 3: Isolate initial planes and bead ---
    initial_plane1_indices = np.where(labels == plane1_label)[0]
    initial_plane2_indices = np.where(labels == plane2_label)[0]
    initial_all_indices = np.arange(len(pcd.points))
    initial_plane_indices = np.concatenate((initial_plane1_indices, initial_plane2_indices))
    initial_bead_indices = np.setdiff1d(initial_all_indices, initial_plane_indices)
    
    # --- Step 4: Refine bead points by filtering based on their average normal ---
    # This step seems to be refining the bead, not the planes as in the previous version.
    # Let's assume the goal is to keep only the core bead points.
    print("Refining bead points by filtering inconsistent normals...")
    avg_normal_bead = np.mean(normals[initial_bead_indices], axis=0)
    avg_normal_bead /= np.linalg.norm(avg_normal_bead)
    dot_products_bead = np.dot(normals[initial_bead_indices], avg_normal_bead)
    angle_threshold_cos = np.cos(np.deg2rad(normal_consistency_angle_deg))
    consistent_mask_bead = dot_products_bead >= angle_threshold_cos
    
    # These are the bead indices after normal consistency refinement
    refined_bead_indices = initial_bead_indices[consistent_mask_bead]
    print(f"Initial bead points: {len(initial_bead_indices)}, after normal refinement: {len(refined_bead_indices)}")

    # The planes remain as they were initially identified
    plane1_pcd = pcd.select_by_index(initial_plane1_indices)
    plane2_pcd = pcd.select_by_index(initial_plane2_indices)
    
    # The bead point cloud before the final blob filtering
    bead_pcd_before_blob_filter = pcd.select_by_index(refined_bead_indices)

    # --- Step 5: NEW - Final cleaning of the bead by keeping only the largest blob ---
    print("Cleaning final bead by keeping only the largest blob...")
    if not bead_pcd_before_blob_filter.has_points():
        print("Warning: No bead points left to clean.")
        return bead_pcd_before_blob_filter, plane1_pcd, plane2_pcd

    # Use DBSCAN on the 3D points to find spatially connected blobs
    blob_labels = np.array(bead_pcd_before_blob_filter.cluster_dbscan(eps=1.5, min_points=20, print_progress=False))
    
    blob_unique_labels, blob_counts = np.unique(blob_labels, return_counts=True)
    
    # Remove noise label (-1) from consideration
    blob_non_noise_mask = blob_unique_labels != -1
    blob_unique_labels = blob_unique_labels[blob_non_noise_mask]
    blob_counts = blob_counts[blob_non_noise_mask]
    
    if len(blob_counts) > 0:
        # Find the label of the largest blob
        largest_blob_label = blob_unique_labels[np.argmax(blob_counts)]
        
        # Select only the points belonging to this largest blob
        largest_blob_mask = blob_labels == largest_blob_label
        final_bead_pcd = bead_pcd_before_blob_filter.select_by_index(np.where(largest_blob_mask)[0])
        print(f"Kept the largest blob with {len(final_bead_pcd.points)} points.")
    else:
        print("Warning: No blobs found in the bead, returning the bead as is.")
        final_bead_pcd = bead_pcd_before_blob_filter

    print(f"Isolated {len(final_bead_pcd.points)} points for the final weld bead.")
    
    return final_bead_pcd, plane1_pcd, plane2_pcd

def contract_to_centerline(bead_pcd, num_iterations=1000, contraction_factor=0.01):
    """
    Contracts a point cloud to its centerline/skeleton using Laplacian smoothing.
    """
    print("Extracting centerline using Laplacian contraction...")
    if not bead_pcd.has_points(): 
        print("Bead point cloud is empty, cannot extract centerline.")
        return o3d.geometry.PointCloud()
        
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


def run(input_pcd):
    bead_pcd, plane1_pcd, plane2_pcd = segment_corner_weld(input_pcd, eps=0.02, min_points=50, normal_consistency_angle_deg=30.0)

    if bead_pcd is None:
        print("Segmentation failed. Exiting.")
        exit()

    # Visualize the segmentation AFTER refinement
    plane1_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Red
    plane2_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Green
    bead_pcd.paint_uniform_color([0.9, 0.5, 0.5]) # Grey
    print("\nVisualizing segmentation AFTER refinement...")
    o3d.visualization.draw_geometries([plane1_pcd, plane2_pcd, bead_pcd], window_name="Segmentation AFTER Refinement")

    # --- Step 2: Clean the isolated bead ---
    print("\nCleaning the isolated bead point cloud...")
    cleaned_bead_pcd, _ = bead_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    cleaned_bead_pcd.paint_uniform_color([0.9, 0.5, 0.5]) # Blue

    # --- Step 3: Extract the centerline from the cleaned bead ---
    centerline_pcd = contract_to_centerline(cleaned_bead_pcd, num_iterations=15000, contraction_factor=0.01)
    centerline_pcd.paint_uniform_color([0, 0, 1]) # Yellow
    
    graph_centerline = clean_and_smooth_centerline(centerline_pcd, num_waypoints=200, k_neighbors=10, voxel_size=0.5)
    graph_centerline.paint_uniform_color([0, 1, 0]) # Blue

    # --- Step 4: Final Visualization ---
    print("\nVisualizing final result...")
    o3d.visualization.draw_geometries(
        [plane1_pcd, plane2_pcd, cleaned_bead_pcd, graph_centerline, centerline_pcd],
        window_name="Final Centerline Extraction",
        width=1000, height=800
    )


if __name__ == "__main__":
    try:
        # Make sure to change this path to your corner weld file
        pcd = o3d.io.read_point_cloud("../data/seam_2/seam_2_seg_6.pcd") 
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Please ensure you have a point cloud file at the specified path.")
        exit()
        
    # --- Step 1: Segment the weld bead from the two main planes ---
    # **Tuning Parameters**:
    # 'eps': For normals, this value is small. 0.05 is a good starting point.
    # 'min_points': The minimum number of points required to form a plane.
    # 'normal_consistency_angle_deg': New parameter for the refinement step.
    bead_pcd, plane1_pcd, plane2_pcd = segment_corner_weld(pcd, eps=0.02, min_points=50, normal_consistency_angle_deg=30.0)

    if bead_pcd is None:
        print("Segmentation failed. Exiting.")
        exit()

    # Visualize the segmentation AFTER refinement
    plane1_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Red
    plane2_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Green
    bead_pcd.paint_uniform_color([0.9, 0.5, 0.5]) # Grey
    print("\nVisualizing segmentation AFTER refinement...")
    o3d.visualization.draw_geometries([plane1_pcd, plane2_pcd, bead_pcd], window_name="Segmentation AFTER Refinement")

    # --- Step 2: Clean the isolated bead ---
    print("\nCleaning the isolated bead point cloud...")
    cleaned_bead_pcd, _ = bead_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    cleaned_bead_pcd.paint_uniform_color([0.9, 0.5, 0.5]) # Blue

    # --- Step 3: Extract the centerline from the cleaned bead ---
    centerline_pcd = contract_to_centerline(cleaned_bead_pcd, num_iterations=15000, contraction_factor=0.01)
    centerline_pcd.paint_uniform_color([0, 0, 1]) # Yellow
    
    graph_centerline = clean_and_smooth_centerline(centerline_pcd, num_waypoints=200, k_neighbors=10, voxel_size=0.5)
    graph_centerline.paint_uniform_color([0, 1, 0]) # Blue

    # --- Step 4: Final Visualization ---
    print("\nVisualizing final result...")
    o3d.visualization.draw_geometries(
        [plane1_pcd, plane2_pcd, cleaned_bead_pcd, graph_centerline, centerline_pcd],
        window_name="Final Centerline Extraction",
        width=1000, height=800
    )
