import open3d as o3d
import numpy as np

def determine_weld_type(pcd, planarity_threshold=0.7, linearity_threshold=0.3):
    """
    Analyzes a point cloud's shape using PCA eigenvalues to determine the weld type.

    This function is designed to be integrated into the main centerline extraction scripts.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        planarity_threshold (float): The value above which a shape is considered planar.
        linearity_threshold (float): The value below which a planar shape is confirmed.
                                     (Note: Corner welds will have high linearity).

    Returns:
        str: A string indicating the weld type: "Flat Plate Weld", "Corner Weld", or "Ambiguous".
    """
    print("Determining weld type...")
    
    points = np.asarray(pcd.points)
    
    if len(points) < 100:
        print("Warning: Point cloud is too small for reliable classification.")
        return "Ambiguous"

    # Calculate covariance and eigenvalues for shape analysis
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues from smallest to largest
    eigenvalues.sort()
    l1, l2, l3 = eigenvalues
    
    # Avoid division by zero for perfectly flat shapes
    if l3 < 1e-9: l3 = 1e-9

    # --- Feature Calculation (New Eigenvalue Ratios) ---
    ratio_l1_l2 = l1 / l2
    ratio_l2_l3 = l2 / l3

    # Calculate planarity and linearity features
    linearity = (l3 - l2) / l3
    planarity = (l2 - l1) / l3
    
    print(f"  - Calculated Planarity: {planarity:.4f}")
    print(f"  - Calculated Linearity: {linearity:.4f}")

    # --- Classification Logic ---
    # This uses the thresholds you've determined from the analysis script.
    # A flat plate is very planar and not very linear.
    # A corner weld is more linear (along the corner) and less planar.
    
    # You can use either the planarity/linearity features or the eigenvalue ratios
    # based on which plot gave you a better separation.
    
    # Example using Planarity and Linearity:
    weld_type = "cylinder" # Default type
    if ratio_l1_l2 < 0.1 and ratio_l2_l3 > 0.3:
        weld_type = "flat"
    elif ratio_l1_l2 > 0.2 and ratio_l2_l3 < 0.2:
        weld_type = "corner"
        
    print(f"==> Determined Weld Type: {weld_type}")
    return weld_type


if __name__ == '__main__':
    # This is an example of how you would use the function in your main scripts.
    
    # 1. Load your point cloud
    try:
        # pcd_to_test = o3d.io.read_point_cloud("./data/seam_1/seam_1_seg_4.pcd") # Example flat weld
        pcd_to_test = o3d.io.read_point_cloud("./data/corner_weld.pcd") # Example corner weld
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    # 2. Call the function to get the weld type
    # ** IMPORTANT **: Replace these threshold values with the ones you determined from the plots!
    weld_type = determine_weld_type(
        pcd_to_test, 
        planarity_threshold=0.7, 
        linearity_threshold=0.3
    )

    # 3. Use the result to choose the correct algorithm
    if weld_type == "flat":
        print("\nExecuting the flat plate centerline extraction script...")
        # Call your flat plate segmentation and centerline functions here
        pass
    elif weld_type == "corner":
        print("\nExecuting the corner weld centerline extraction script...")
        # Call your corner weld segmentation and centerline functions here
        pass
    else:
        print("\nWeld type is ambiguous. Manual inspection may be required.")
        pass
