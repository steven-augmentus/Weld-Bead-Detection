import numpy as np
import open3d as o3d

def _principal_axis(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Point cloud is empty")
    mean = pts.mean(axis=0)
    X = pts - mean
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)  # eigh since C is symmetric
    order = np.argsort(vals)[::-1]
    v = vecs[:, order[0]]
    # Deterministic direction
    if v[2] < 0: v = -v
    return mean, v / (np.linalg.norm(v) + 1e-12)

def _moving_average(arr: np.ndarray, window: int):
    window = max(1, int(window))
    if window <= 1 or arr.shape[0] < 3:
        return arr
    if window % 2 == 0:
        window += 1
    pad = window // 2
    pad_front = arr[0:1].repeat(pad, axis=0)
    pad_back = arr[-1:].repeat(pad, axis=0)
    padded = np.vstack([pad_front, arr, pad_back])
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, padded)
    return smoothed

def _make_lineset(points: np.ndarray) -> o3d.geometry.LineSet:
    if points.shape[0] < 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.empty((0,2), dtype=np.int32))
        ls.paint_uniform_color([1.0, 0.2, 0.0])
        return ls
    lines = np.array([[i, i + 1] for i in range(points.shape[0] - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([1.0, 0.2, 0.0])
    return ls

def compute_centerline(
    pcd: o3d.geometry.PointCloud,
    n_slices: int = 60,
    min_points: int = 30,
    smooth_window: int = 10,
    force_endpoints: bool = True, 
):
    """
    Slice the cloud along its principal axis and take the centroid of each slice.
    Optionally smooth with a short moving average.
    Returns (centerline_points (K,3) ndarray, o3d.geometry.LineSet).
    """
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError("pcd must be an open3d.geometry.PointCloud")

    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError("Not enough points to compute a centerline")

    mean, axis = _principal_axis(pcd)

    # 1D coordinate along principal axis
    t = (pts - mean) @ axis
    tmin, tmax = float(t.min()), float(t.max())

    if tmax - tmin < 1e-6:
        cl = mean.reshape(1, 3)
        return cl, _make_lineset(cl)

    # Bin edges for slices
    n_slices = max(5, int(n_slices))
    edges = np.linspace(tmin, tmax, n_slices + 1)

    centers = []
    for i in range(n_slices):
        lo, hi = edges[i], edges[i + 1]
        mask = (t >= lo) & (t < hi) if i < n_slices - 1 else (t >= lo) & (t <= hi)
        idx = np.flatnonzero(mask)
        if idx.size >= min_points:
            centers.append(pts[idx].mean(axis=0))

    if len(centers) < 2:
        # Fallback: orthogonal least-squares line fit through all points (PCA line)
        cl = np.vstack([mean - axis * (tmax - tmin) * 0.5, mean + axis * (tmax - tmin) * 0.5])
        return cl, _make_lineset(cl)

    cl = np.vstack(centers)

    # --- NEW bit: enforce spanning full length ---
    if force_endpoints:
        # extreme slice means (even if they had < min_points)
        first_slice = pts[t.argmin()]
        last_slice  = pts[t.argmax()]
        # prepend/append if they’re not already close
        if np.linalg.norm(cl[0] - first_slice) > 1e-6:
            cl = np.vstack([first_slice, cl])
        if np.linalg.norm(cl[-1] - last_slice) > 1e-6:
            cl = np.vstack([cl, last_slice])

    if smooth_window and smooth_window > 1:
        cl = _moving_average(cl, smooth_window)

    return cl, _make_lineset(cl)

def visualize_with_centerline(
    pcd: o3d.geometry.PointCloud,
    centerline_points: np.ndarray,
    point_color=(0.6, 0.6, 0.6),
):
    # Copy (Open3D 0.18+: clone(); older: construct new)
    pcd_copy = pcd.clone() if hasattr(pcd, "clone") else o3d.geometry.PointCloud(pcd)
    pcd_copy.paint_uniform_color(point_color)
    ls = _make_lineset(centerline_points)
    # Optional spheres on waypoints for legibility
    spheres = []
    if centerline_points.shape[0] > 0:
        bbox = pcd_copy.get_axis_aligned_bounding_box()
        diag = float(np.linalg.norm(bbox.get_extent()))
        r = max(1e-3, diag * 0.005)
        step = max(1, centerline_points.shape[0] // 30)
        for p in centerline_points[::step]:
            m = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            m.translate(p)
            m.compute_vertex_normals()
            m.paint_uniform_color([1.0, 0.2, 0.0])
            spheres.append(m)
    o3d.visualization.draw_geometries([pcd_copy, ls, *spheres], window_name="Point Cloud + Centerline")