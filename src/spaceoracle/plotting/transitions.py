import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler

from .layout import plot_quiver, plot_vectorfield
from .shift import *


def estimate_transitions_2D(adata, delta_X, embedding, annot=None, normalize=True, 
n_neighbors=200, vector_scale=1, n_jobs=1):

    P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    V_simulated = project_probabilities(P, embedding, normalize=normalize)

    grid_scale = 10 / np.mean(abs(np.diff(embedding)))
    print(grid_scale)
    get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, 
                                                           int((max_val - min_val + 1) * grid_scale))

    grid_x = get_grid_points(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    grid_y = get_grid_points(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
    size_x, size_y = len(grid_x), len(grid_y)
    
    vector_field = np.zeros((size_x, size_y, 2))

    x_thresh = (grid_x[1] - grid_x[0]) / 2
    y_thresh = (grid_y[1] - grid_y[0]) / 2

    get_neighborhood = lambda grid_point, embedding: np.where(
        (np.abs(embedding[:, 0] - grid_point[0]) <= x_thresh) &  
        (np.abs(embedding[:, 1] - grid_point[1]) <= y_thresh)   
    )[0]

    for idx, grid_point in tqdm(enumerate(grid_points), desc='Computing vectors', total=len(grid_points)):

        # Get average vector within neighborhood
        indices = get_neighborhood(grid_point, embedding)
        if len(indices) <= 0:
            continue
        nbr_vector = np.mean(V_simulated[indices], axis=0)
        nbr_vector *= len(indices)       # upweight vectors with lots of cells
            
        grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
        vector_field[grid_idx_x, grid_idx_y] = nbr_vector
    
    vector_field = vector_field.reshape(-1, 2)
    
    ### for testing, delete or save properly later
    # adata.uns['grid_points'] = grid_points
    # adata.uns['vector_field'] = vector_field
    adata.uns['nn_transition_P'] = P
    # adata.uns['V_simulated'] = V_simulated
    ###

    vector_scale = vector_scale / np.max(vector_field)
    vector_field *= vector_scale
    if annot is None:
        background = None
    else:
        background = {
            'X': embedding[:, 0], 
            'Y': embedding[:, 1], 
            'annot': list(adata.obs[annot]),
        }

    plot_quiver(grid_points, vector_field, background=background)


def estimate_transitions_3D(adata, delta_X, embedding, annot=None, normalize=True, 
vector_scale=0.1, grid_scale=1, n_jobs=1):
    
    P = estimate_transition_probabilities(adata, delta_X, embedding, n_jobs=n_jobs)
    V_simulated = project_probabilities(P, embedding, normalize)
    
    get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, 
                                                           int((max_val - min_val + 1) * grid_scale))

    grid_x = get_grid_points(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    grid_y = get_grid_points(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    grid_z = np.unique(embedding[...,2])
    # grid_z = [0.5]

    grid_points = np.array(np.meshgrid(grid_x, grid_y, grid_z)).T.reshape(-1, 3)
    size_x, size_y, size_z = len(grid_x), len(grid_y), len(grid_z)
    vector_field = np.zeros((size_x, size_y, size_z, 3))

    x_thresh = (grid_x[1] - grid_x[0]) / 2
    y_thresh = (grid_y[1] - grid_y[0]) / 2
    z_thresh = 0.1
    # z_thresh = 1

    get_neighborhood = lambda grid_point, embedding: np.where(
        (np.abs(embedding[:, 0] - grid_point[0]) <= x_thresh) &  
        (np.abs(embedding[:, 1] - grid_point[1]) <= y_thresh) &
        (np.abs(embedding[:, 2] - grid_point[2]) <= z_thresh)
    )[0]
    

    for idx, grid_point in tqdm(enumerate(grid_points), desc='Computing vectors', total=len(grid_points)):
        smoothed_vector = np.zeros(3)

        # Get average vector within neighborhood
        indices = get_neighborhood(grid_point, embedding)
        if len(indices) <= 0:
            continue
        nbr_vector = np.mean(V_simulated[indices], axis=0)
        nbr_vector *= len(indices)       # upweight vectors with lots of cells
        
        grid_idx_x, grid_idx_y, grid_idx_z = np.unravel_index(idx, (size_x, size_y, size_z))
        vector_field[grid_idx_x, grid_idx_y, grid_idx_z] = nbr_vector
    
    ### for testing, delete or save properly later
    # adata.uns['grid_points'] = grid_points
    adata.uns['nn_transition_P'] = P
    # adata.uns['P'] = P
    # adata.uns['V_simulated'] = V_simulated
    ###

    vector_field = vector_field * vector_scale
    
    if annot is not None:
        background = pd.DataFrame({
                'X': adata.obsm['spatial_3D'][:, 0],
                'Y': adata.obsm['spatial_3D'][:, 1],
                'Z': adata.obsm['spatial_3D'][:, 2],
                'annot': adata.obs[annot]
            })
    else:
        background = None

    plot_vectorfield(grid_points, vector_field, background)


def estimate_celltocell_transitions(adata, delta_X, embedding, cluster=None, annot=None, log_P=True, n_jobs=1):

    n_neighbors=200

    if cluster is not None:
        adata = adata.copy()
        cell_idxs = np.where(adata.obs[annot] == cluster)[0]

        delta_X = delta_X[cell_idxs, :]
        embedding = embedding[cell_idxs, :]
        adata = adata[adata.obs[annot] == cluster]

        P = estimate_transition_probabilities(
            adata, delta_X, embedding, n_neighbors=n_neighbors, random_neighbors=True, n_jobs=n_jobs)

    elif 'transition_P' not in adata.uns:
        # this it taking way too long
        # P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=None, n_jobs=n_jobs)
        
        # quicker alternative, although may need adjusting
        P = estimate_transition_probabilities(
            adata, delta_X, embedding, n_neighbors=200, random_neighbors=True, n_jobs=n_jobs)
        adata.uns['transition_P'] = P
    
    else:
        P = adata.uns['transition_P']

    x = embedding[:, 0]
    y = embedding[:, 1]

    if log_P:
        P = np.where(P != 0, np.log(P), 0)

    intensity = np.sum(P, axis=0).reshape(-1, 1)
    intensity = MinMaxScaler().fit_transform(intensity)

    plt.scatter(x, y, c=intensity, cmap='coolwarm', s=1, alpha=0.9, label='Transition Probabilities')

    plt.colorbar(label='Transition Odds Post-perturbation')
    if cluster is not None:
        plt.title(f'{cluster} Subset Transition Probabilities')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.tight_layout()


def estimate_celltype_transitions(adata, delta_X, embedding, annot='rctd_cluster', n_neighbors=200, vector_scale=100,
                        visual_clusters=['B-cell', 'Th2', 'Cd8 T-cell'], n_jobs=1):
    
    missing_clusters = set(visual_clusters) - set(adata.obs[annot])
    if missing_clusters:
        raise ValueError(f"Invalid cell types: {', '.join(missing_clusters)}")

    P = estimate_transition_probabilities(
        adata, delta_X, embedding, n_neighbors=n_neighbors, annot=annot, 
        random_neighbors='even', n_jobs=n_jobs
    )

    # Convert cell x cell -> cell x cell-type transition P
    unique_clusters, cluster_indices = np.unique(adata.obs[annot], return_inverse=True)
    cluster_mask = np.zeros((adata.n_obs, len(unique_clusters)))
    cluster_mask[np.arange(adata.n_obs), cluster_indices] = 1

    P_ct = P @ cluster_mask

    # Renormalize after selecting cell types of interest
    visual_idxs = [unique_clusters.tolist().index(ct) for ct in visual_clusters]
    P_ct = P_ct[:, visual_idxs]
    # if renormalize:
    #     P_ct = P_ct / P_ct.sum(axis=1, keepdims=True)

    # Project probabilities into vectors for each cell
    angles = np.linspace(0, 360, len(visual_clusters), endpoint=False)
    angles_rad = np.deg2rad(angles)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)

    directions = np.column_stack((x, y)) # (ct x 2)
    vectors = P_ct @ directions          # (cell x 2)
    adata.obsm['celltype_vectors'] = vectors

    # x_positions = adata.obsm['spatial'][:, 0]
    # y_positions = adata.obsm['spatial'][:, 1]
    x_positions = embedding[:, 0]
    y_positions = embedding[:, 1]

    vectors = vectors * vector_scale
    x_directions = vectors[:, 0]
    y_directions = vectors[:, 1]

    # Plot gray scatter
    categories = adata.obs[annot].astype('category')
    codes = categories.cat.codes
    plt.scatter(x_positions, y_positions, color='grey', alpha=0.3, s=3)

    # Plot quiver
    max_indices = np.argmax(P_ct, axis=1)
    colors = np.array([codes[i] for i in max_indices])
    # highest_transition = np.argmax(P, axis=1)
    # colors = [codes[i] for i in highest_transition]

    cmap = cm.get_cmap('tab10')
    rgb_values = [cmap(c)[:3] for c in colors]

    plt.quiver(x_positions, y_positions, x_directions, y_directions, color=rgb_values, 
            scale=1, angles="xy", scale_units="xy", linewidth=0.15)

    # Plot colored scatter
    # scatter = plt.scatter(x_positions, y_positions, c=codes, alpha=0.8, s=3, cmap='tab10', edgecolors='none')
    # handles, labels = scatter.legend_elements(num=len(unique_clusters))
    # plt.legend(handles, unique_clusters, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Place quiver anchors
    anchor_offset = 300
    for i, (dx, dy, label) in enumerate(zip(directions[:, 0], directions[:, 1], visual_clusters)):
        anchor_x = dx * anchor_offset
        anchor_y = dy * anchor_offset
        plt.quiver(0, 0, anchor_x, anchor_y, color=cmap(i), angles="xy", scale_units="xy", scale=1, width=0.005)
        plt.text(anchor_x * 2.1, anchor_y * 1.9, label, color=cmap(i), ha='center', va='center', fontsize=10)

    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()