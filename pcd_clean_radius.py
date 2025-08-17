# pcd_clean_radius.py
import numpy as np
import open3d as o3d

def estimate_radius(pcd: o3d.geometry.PointCloud, k: int = 3, sample: int = 3000, factor: float = 2.5) -> float:
    """
    Heuristic radius ~ factor * median distance to the k-th nearest neighbor.
    Keeps things scale-aware without DBSCAN.
    """
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n == 0:
        return 0.01
    k = max(1, int(k))
    sample = min(n, int(sample))
    rng = np.random.default_rng(0)
    idxs = rng.choice(n, size=sample, replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    d = []
    for i in idxs:
        # Returns squared distances including the point itself; request k+1 to skip self
        _, _, dist2 = tree.search_knn_vector_3d(pcd.points[i], k + 1)
        if len(dist2) > k:
            d.append(float(dist2[-1]) ** 0.5)
    if not d:
        return 0.01
    return float(np.median(d)) * float(factor)

class _DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def connected_components_radius(pcd: o3d.geometry.PointCloud, radius: float) -> np.ndarray:
    """
    Compute connected components by linking points that are within `radius` of each other.
    Returns labels array in [0..C-1] for components, or empty array if no points.
    """
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    radius = float(max(1e-9, radius))
    tree = o3d.geometry.KDTreeFlann(pcd)
    dsu = _DSU(n)
    # For each point, link to neighbors with j>i to avoid duplicate unions
    for i in range(n):
        _, idxs, _ = tree.search_radius_vector_3d(pcd.points[i], radius)
        for j in idxs:
            if j > i:
                dsu.union(i, j)
    # Relabel components to 0..C-1
    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int32)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)

def _select_indices_by_min_size(labels: np.ndarray, min_component_size: int):
    keep = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if idx.size >= int(min_component_size):
            keep.append(idx)
    if not keep:
        return np.array([], dtype=np.int32)
    return np.concatenate(keep).astype(np.int32)

def keep_largest_component_radius(pcd: o3d.geometry.PointCloud, radius: float, min_component_size: int = 0) -> o3d.geometry.PointCloud:
    """Keep only the largest radius-connected component. If min_component_size>0, require that size; else return largest regardless."""
    labels = connected_components_radius(pcd, radius)
    if labels.size == 0:
        return pcd
    # sizes per label
    uniq, counts = np.unique(labels, return_counts=True)
    # optionally ignore components below threshold
    if min_component_size > 0:
        mask = counts >= int(min_component_size)
        if not np.any(mask):  # none pass threshold; return empty
            return o3d.geometry.PointCloud()
        uniq, counts = uniq[mask], counts[mask]
    largest = int(uniq[np.argmax(counts)])
    idx = np.flatnonzero(labels == largest).tolist()
    return pcd.select_by_index(idx)

def remove_components_smaller_than_radius(pcd: o3d.geometry.PointCloud, radius: float, min_component_size: int) -> o3d.geometry.PointCloud:
    """Remove any radius-connected components with < min_component_size points."""
    if min_component_size <= 0:
        return pcd
    labels = connected_components_radius(pcd, radius)
    if labels.size == 0:
        return pcd
    keep_idx = _select_indices_by_min_size(labels, min_component_size)
    if keep_idx.size == 0:
        return o3d.geometry.PointCloud()
    return pcd.select_by_index(keep_idx.tolist())

def colorize_by_label(pcd: o3d.geometry.PointCloud, labels: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a copy colored by component label."""
    pts = np.asarray(pcd.points)
    out = o3d.geometry.PointCloud()
    if pts.size == 0:
        return out
    out.points = o3d.utility.Vector3dVector(pts.copy())
    colours = np.zeros((pts.shape[0], 3), dtype=float)
    uniq = np.unique(labels) if labels.size > 0 else []
    for u in uniq:
        np.random.seed(int(u) + 123)
        colours[labels == u] = np.random.rand(3) * 0.7 + 0.3
    out.colors = o3d.utility.Vector3dVector(colours)
    return out
