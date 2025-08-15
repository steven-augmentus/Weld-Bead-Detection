import open3d as o3d
import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import entropy
import matplotlib.pyplot as plt
from utils.centerline2graph import clean_and_smooth_centerline
from utils.weld_type_cls import determine_weld_type
from utils import flat_bead
from utils import corner_bead
from utils import cylinder_bead

def create_coordinate_frame(size=10.0):
    """Creates an Open3D LineSet object representing a coordinate frame."""
    lines = [
        [0, 1], [0, 2], [0, 3],
        [1, 4], [2, 5], [3, 6]
    ]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    
    points = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [0, size, 0],
        [0, 0, size],
        [size, size, size],
        [size, size, 0],
        [size, 0, size]
    ])
    
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset

if __name__ == "__main__":
    # Time the execution of the script
    import time
    start_time = time.time()
    try:
        pcd = o3d.io.read_point_cloud("./data/08.cropped.pcd")
        # Downsample the point cloud for faster processing
        pcd = pcd.voxel_down_sample(voxel_size=0.5)
        weld_type = determine_weld_type(pcd)
        print(f"Detected Weld Type: {weld_type}")
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    #print("\n--- Initial Visualization ---")
    #o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    
    if weld_type == "corner":
        print("\n--- Processing Corner Weld ---")
        corner_bead.run(pcd)
    elif weld_type == "cylinder":
        print("\n--- Processing Cylinder Weld ---")
        cylinder_bead.run(pcd)
    elif weld_type == "flat":
        print("\n--- Processing Flat Weld ---")
        flat_bead.run(pcd)
    else:
        print("\n--- Unsupported Weld Type ---")
        exit()
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
        