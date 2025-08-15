import open3d as o3d
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import splprep, splev

def clean_and_smooth_centerline(pcd_centerline, num_waypoints=100, k_neighbors=10, voxel_size=0.5):
    """
    Cleans a raw centerline point cloud by removing branches and outliers, then
    smooths it by fitting a B-spline and resampling evenly spaced waypoints.

    Args:
        pcd_centerline (o3d.geometry.PointCloud): The raw centerline point cloud,
                                                  which may contain branches or loops.
        num_waypoints (int): The desired number of evenly spaced points in the final centerline.
        k_neighbors (int): The number of nearest neighbors to use when constructing the graph.
        voxel_size (float): The size of the voxel for downsampling. This is used to
                            thin out dense areas like junctions.

    Returns:
        o3d.geometry.PointCloud: A new point cloud representing the final, cleaned,
                                 and smoothed waypoints in order.
    """
    print("Cleaning and smoothing centerline...")
    points = np.asarray(pcd_centerline.points)
    
    if len(points) < k_neighbors:
        print("Warning: Not enough points to clean centerline. Returning original.")
        return pcd_centerline

    # --- NEW: Voxel Downsampling to thin dense areas ---
    # This is the key step to simplify complex junctions before pathfinding.
    print(f"  - Original point count: {len(points)}")
    thinned_pcd = pcd_centerline.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(thinned_pcd.points)
    print(f"  - Thinned point count: {len(points)}")

    if len(points) < k_neighbors:
        print("Warning: Not enough points after downsampling. Try a smaller voxel_size.")
        return pcd_centerline

    # --- 1. Build a k-NN graph from the thinned points ---
    pcd_tree = o3d.geometry.KDTreeFlann(thinned_pcd)
    graph_rows = []
    graph_cols = []
    graph_data = []
    
    for i in range(len(points)):
        [k, idx, dists] = pcd_tree.search_knn_vector_3d(points[i], k_neighbors)
        graph_rows.extend([i] * (k - 1))
        graph_cols.extend(idx[1:]) # Exclude self
        graph_data.extend(dists[1:])

    # Create a SciPy sparse matrix for the graph
    graph = csr_matrix((graph_data, (graph_rows, graph_cols)), shape=(len(points), len(points)))

    # --- 2. Find the two most distant points within the largest connected component ---
    dist_matrix = shortest_path(csgraph=graph, directed=False)
    
    finite_dist_matrix = dist_matrix.copy()
    finite_dist_matrix[np.isinf(finite_dist_matrix)] = -1 # Ignore infinite distances
    
    start_node, end_node = np.unravel_index(np.argmax(finite_dist_matrix), finite_dist_matrix.shape)
    
    print(f"  - Found start node {start_node} and end node {end_node} within a connected component.")

    # --- 3. Find the shortest path between these two nodes (removes branches) ---
    _, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_node, return_predecessors=True)
    
    path = []
    current_node = end_node
    while current_node != -9999 and current_node != start_node: # Loop until we reach the start or an invalid predecessor
        path.append(current_node)
        current_node = predecessors[current_node]
    path.append(start_node) # Add the start node to complete the path
    
    path.reverse() # Path is constructed backwards, so reverse it
    
    print(f"  - Length of cleaned path before smoothing: {len(path)} points.")
    
    # Get the 3D points for the cleaned path
    cleaned_points = points[path]
    
    if len(cleaned_points) < 4: # Spline fitting needs at least 4 points
        print("Warning: Path is still too short for spline fitting after cleaning. Returning cleaned points.")
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        return cleaned_pcd

    # --- 4. Fit a B-spline to the cleaned path for smoothing ---
    tck, u = splprep([cleaned_points[:, 0], cleaned_points[:, 1], cleaned_points[:, 2]], s=2)
    
    # --- 5. Resample evenly spaced waypoints from the spline ---
    u_new = np.linspace(u.min(), u.max(), num_waypoints)
    x_new, y_new, z_new = splev(u_new, tck)
    
    smoothed_points = np.vstack((x_new, y_new, z_new)).T
    
    # --- 6. Return the final point cloud ---
    final_centerline_pcd = o3d.geometry.PointCloud()
    final_centerline_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    
    print("Centerline cleaning and smoothing complete.")
    return final_centerline_pcd


if __name__ == '__main__':
    # --- Create a sample noisy centerline with a branch to test the function ---
    print("--- Testing Centerline Cleaning Function ---")
    
    # Main line
    main_line = np.array([[i, np.sin(i/5.0) * 2, 0] for i in np.linspace(0, 20, 50)])
    main_line += np.random.rand(*main_line.shape) * 0.5
    
    # Branch
    branch = np.array([[10 + i, (np.sin(10/5.0) * 2) + i*2, 0] for i in np.linspace(0, 3, 15)])
    branch += np.random.rand(*branch.shape) * 0.5

    # Dense junction
    junction = np.random.rand(50, 3) * 2 + np.array([10, np.sin(2)*2, 0]) - 1
    
    # Combine into a single point cloud
    raw_points = np.vstack((main_line, branch, junction))
    raw_centerline_pcd = o3d.geometry.PointCloud()
    raw_centerline_pcd.points = o3d.utility.Vector3dVector(raw_points)
    raw_centerline_pcd.paint_uniform_color([1, 0, 0]) # Red for raw
    
    print("Visualizing raw, noisy centerline (in red)...")
    o3d.visualization.draw_geometries([raw_centerline_pcd], window_name="Raw Centerline")

    # --- Run the cleaning and smoothing function ---
    final_centerline_pcd = clean_and_smooth_centerline(raw_centerline_pcd, num_waypoints=100, voxel_size=0.8)
    final_centerline_pcd.paint_uniform_color([0, 1, 0]) # Green for final
    
    # --- Visualize the final result ---
    print("Visualizing final, cleaned centerline (in green) overlaid on raw (in red)...")
    
    points_final = np.asarray(final_centerline_pcd.points)
    lines_final = [[i, i + 1] for i in range(len(points_final) - 1)]
    colors_final = [[0, 1, 0] for i in range(len(lines_final))]
    line_set_final = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_final),
        lines=o3d.utility.Vector2iVector(lines_final),
    )
    line_set_final.colors = o3d.utility.Vector3dVector(colors_final)

    o3d.visualization.draw_geometries([raw_centerline_pcd, line_set_final], window_name="Final vs. Raw Centerline")
