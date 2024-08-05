from numba import jit
from tqdm import tqdm
import numpy as np


@jit
def generate_grid_centers(m, n, xmin, xmax, ymin, ymax):
    centers = []
    cell_width = (xmax - xmin) / n
    cell_height = (ymax - ymin) / m
    
    for i in range(m):
        for j in range(n):
            x = xmin + (j + 0.5) * cell_width
            y = ymax - (i + 0.5) * cell_height
            centers.append((x, y))    
    return centers

@jit
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def xy2spatial(x, y, m, n, dist_fn = distance):
    assert len(x) == len(y)
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xy = np.column_stack([x, y]).astype(float)
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    spatial_maps = np.zeros((len(x), m, n))
    with tqdm(total=len(xy), disable=True) as pbar:
        for s, coord in enumerate(xy):
            spatial_maps[s] = np.array([dist_fn(coord, c) for c in centers]).reshape(m, n)
            pbar.update()
        
    return spatial_maps

def same_cell(point1, point2, cutoff = 120):
    x1, y1 = point1
    x2, y2 = point2
    if abs(x1 - x2) + abs(y1 - y2) < cutoff:
        return 1
    return 0

def xyc2spatial(xyc, m, n):
    x, y, c = xyc.T
    spatial_maps = xy2spatial(x, y, m, n)

    clusters = np.unique(c)
    c_spatial_maps = [] # (cluster x cells x m x n)
    for cluster in clusters:
        mask = xyc[:, -1] == cluster
        cluster_coords = xyc[mask]
        cx, cy, _ = cluster_coords.T

        cluster_mask = xy2spatial(cx, cy, m, n, dist_fn=same_cell)
        cluster_mask = np.max(cluster_mask, axis = 0)

        c_spatial_maps.append(cluster_mask * spatial_maps)

    return np.array(c_spatial_maps).transpose(1, 2, 3, 0)