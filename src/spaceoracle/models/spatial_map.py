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


def xy2spatial(x, y, m, n):
    assert len(x) == len(y)
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xy = np.column_stack([x, y]).astype(float)
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    spatial_maps = np.zeros((len(x), m, n))
    with tqdm(total=len(xy), disable=True) as pbar:
        for s, coord in enumerate(xy):
            spatial_maps[s] = np.array([distance(coord, c) for c in centers]).reshape(m, n)
            pbar.update()
        
    return spatial_maps