from numba import jit
from tqdm import tqdm
import numpy as np

from scipy.spatial import KDTree


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

##TODO: combine this with xy2spatial
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

def xy2distance(coords, n_top):

    tree = KDTree(coords)
    distances = []
    for point in coords:
        dist, indices = tree.query(point, k=n_top + 1)
        distances.append(dist[1:])

    return np.array(distances)