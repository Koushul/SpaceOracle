from numba import jit
from tqdm import tqdm
import numpy as np
from ..tools.utils import deprecated

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


def xyc2spatial(x, y, c, m, n, split_channels=True, disable_tqdm=True):
    
    assert len(x) == len(y) == len(c)
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xyc = np.column_stack([x, y, c]).astype(float)
    
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    clusters = np.unique(c).astype(int)
    
    spatial_maps = np.zeros((len(x), m, n))
    mask = np.zeros((len(clusters), m, n))
    with tqdm(total=len(xyc), disable=disable_tqdm, desc='🌍️ Generating spatial maps') as pbar:
        
        for s, coord in enumerate(xyc):
            x_, y_, cluster = coord
            
            dist_map = np.array([distance((x_, y_), c) for c in centers]).reshape(m, n)
            
            nearest_center_idx = np.argmin(dist_map)
            u, v = np.unravel_index(nearest_center_idx, (m, n))
            mask[int(cluster)][u, v] = 1

            spatial_maps[s] = dist_map
            
            pbar.update()
    
    
    spatial_maps = np.repeat(np.expand_dims(spatial_maps, axis=1), len(clusters), axis=1)
    mask = np.repeat(np.expand_dims(mask, axis=0), spatial_maps.shape[0], axis=0)

    # channel_wise_maps = spatial_maps*mask 
    channel_wise_maps = (1.0/spatial_maps)*mask 
    

        
    assert channel_wise_maps.shape == (len(x), len(clusters), m, n)
    
    if split_channels:
        return channel_wise_maps
    else:
        return channel_wise_maps.sum(axis=1)
    
    