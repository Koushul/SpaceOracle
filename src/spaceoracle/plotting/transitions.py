import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns 

import umap
from sklearn.preprocessing import MinMaxScaler
from numba import jit

from .layout import plot_quiver, get_quiver_grid
from .shift import *


def estimate_transitions_2D(adata, delta_X, embedding, annot=None, normalize=True, 
n_neighbors=200, grid_scale=1, vector_scale=1, n_jobs=1, savepath=False):

    P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    V_simulated = project_probabilities(P, embedding, normalize=normalize)

    grid_points, vector_field = get_quiver_grid(embedding, grid_scale, V_simulated)

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
    plot_quiver(grid_points, vector_field, background=background, savepath=savepath)

def randomize_transitions_2D(adata, delta_X, embedding, annot=None, normalize=True, 
n_neighbors=200, grid_scale=1, vector_scale=0.1, n_jobs=1, savepath=False):

    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    estimate_transitions_2D(adata, delta_X_rndm , embedding, annot, normalize, 
        n_neighbors, grid_scale, vector_scale, n_jobs, savepath)


def estimate_celltype_transitions(adata, delta_X, embedding, annot='rctd_cluster', n_neighbors=200, vector_scale=100,
                        visual_clusters=['B-cell', 'Th2', 'Cd8 T-cell'], grid_scale=1, n_jobs=1, savepath=False):
    
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

    grid_points, vectors = get_quiver_grid(embedding, grid_scale, vectors)
    vector_scale = vector_scale / np.max(vectors)
    vectors *= vector_scale

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    background = {
            'X': embedding[:, 0], 
            'Y': embedding[:, 1], 
            'annot': list(adata.obs[annot]),
        }
    plot_quiver(grid_points, vectors, background, ax=ax, savepath=savepath)

    # Place quiver anchors
    anchor_offset = 300
    for dx, dy, label in zip(directions[:, 0], directions[:, 1], visual_clusters):
        anchor_x = dx * anchor_offset
        anchor_y = dy * anchor_offset
        plt.quiver(0, 0, anchor_x, anchor_y, angles="xy", scale_units="xy", scale=1, width=0.005)
        plt.text(anchor_x * 2.1, anchor_y * 1.9, label, ha='center', va='center', fontsize=10)

    plt.axis('off')
    plt.axis('equal')
    plt.show()


def estimate_celltocell_transitions(adata, delta_X, embedding, cluster=None, annot=None, log_P=True, n_jobs=1):

    n_neighbors=200

    # Recompute transition probabilities for cluster subset
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
    

    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    P_null = estimate_transition_probabilities(
        adata, delta_X_rndm, embedding, n_neighbors=n_neighbors, random_neighbors=True, n_jobs=n_jobs
    )

    x = embedding[:, 0]
    y = embedding[:, 1]

    if log_P:
        P = np.where(P != 0, np.log(P), 0)
        P_null = np.where(P_null != 0, np.log(P_null), 0)
    
    P = P - P_null
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


def contour_shift(adata_train, seed=1334, savepath=False):

    # Load data
    perturbed = adata_train.layers['simulated_count']
    gex = adata_train.layers['imputed_count']

    # Create UMAP embeddings
    reducer = umap.UMAP(random_state=seed, n_neighbors=50, min_dist=1.0, spread=5.0)
    X = np.vstack([gex, perturbed])
    umap_coords = reducer.fit_transform(X)

    # Split coordinates back into WT and KO
    n_wt = gex.shape[0]
    wt_umap = umap_coords[:n_wt]
    ko_umap = umap_coords[n_wt:]

    # Create elegant UMAP visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot cell type scatter points with custom styling
    sns.scatterplot(
        x=wt_umap[:,0], 
        y=wt_umap[:,1],
        hue=adata_train.obs.rctd_celltypes.values,
        alpha=0.5,
        s=20,
        style=adata_train.obs.rctd_celltypes.values,
        ax=ax,
        markers=['o', 'X', '<', '^', 'v', 'D', '>'],
    )

    # Add density contours for WT and KO
    for coords, label, color in [(wt_umap, 'WT', 'grey'), 
                                (ko_umap, 'KO', 'black')]:
        sns.kdeplot(
            x=coords[:,0],
            y=coords[:,1], 
            levels=8,
            alpha=1,
            linewidths=2,
            label=label,
            color=color,
            ax=ax,
            legend=True
        )

    # Style the plot
    ax.set_title('Cell Identity Shift after Knockout', pad=20, fontsize=12)
    ax.set_xlabel('UMAP 1', labelpad=10)
    ax.set_ylabel('UMAP 2', labelpad=10)
    ax.legend(ncol=1, loc='upper left', frameon=False)

    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, transparent=True)
    plt.show()





# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
