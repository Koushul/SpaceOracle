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

@deprecated('Please use the xyc2spatial function instead.')
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


def xyc2spatial(x, y, c, m, n, split_channels=True, disable_tqdm=True):
    
    ##FIX ME: This is not correct

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
    
    
@deprecated('Please use the xyc2spatial function instead.')
def cluster_masks(x, y, c, m, n):
    assert len(x) == len(y) == len(c)
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xyc = np.column_stack([x, y, c]).astype(float)
    
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    clusters = np.unique(c).astype(int)
    
    cluster_mask = np.zeros((len(clusters), m, n))
    
    
    with tqdm(total=len(xyc), disable=False, desc='Generating spatial cluster maps') as pbar:
        
        for s, coord in enumerate(xyc):
            x, y, cluster = coord
            distances = np.array([distance([x, y], center) for center in centers]).reshape(m, n)
            nearest_center_idx = np.argmin(distances)
            u, v = np.unravel_index(nearest_center_idx, (m, n))

            cluster_mask[int(cluster)][u, v] = 1
            
            pbar.update()
        
        
    return cluster_mask

@jit
def apply_masks_to_images(images, masks):
    num_images, img_height, img_width = images.shape
    num_masks, mask_height, mask_width = masks.shape
    
    assert img_height == mask_height and img_width == mask_width
    
    output = np.zeros((num_images, num_masks, img_height, img_width))
    
    for i in range(num_images):
        for j in range(num_masks):
            output[i, j] = images[i] * masks[j]
    
    return output