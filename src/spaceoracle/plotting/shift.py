import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from .transitions import estimate_transition_probabilities

def estimate_transitions(adata, delta_X, embedding, annot='rctd_cluster', n_neighbors=200, vector_scale=1,
                        visual_clusters=['B-cell', 'Th2', 'Cd8 T-cell'], renormalize=False, n_jobs=1):
    
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
    if renormalize:
        P_ct = P_ct / P_ct.sum(axis=1, keepdims=True)

    # Project probabilities into vectors for each cell
    angles = np.linspace(0, 360, len(visual_clusters), endpoint=False)
    angles_rad = np.deg2rad(angles)
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)

    directions = np.column_stack((x, y)) # (ct x 2)
    vectors = P_ct @ directions          # (cell x 2)

    x_positions = adata.obsm['spatial'][:, 0]
    y_positions = adata.obsm['spatial'][:, 1]

    vectors = vectors * vector_scale
    x_directions = vectors[:, 0]
    y_directions = vectors[:, 1]

    # Plot 
    categories = adata.obs[annot].astype('category')
    codes = categories.cat.codes

    # Plot quiver
    max_indices = np.argmax(P_ct, axis=1)
    colors = np.array([codes[i] for i in max_indices])
    # highest_transition = np.argmax(P, axis=1)
    # colors = [codes[i] for i in highest_transition]

    cmap = cm.get_cmap('tab10')
    rgb_values = [cmap(c)[:3] for c in colors]
    plt.quiver(x_positions, y_positions, x_directions, y_directions, color=rgb_values, 
               scale=0.01, angles="xy", scale_units="xy", linewidth=0.15)

    # Plot scatter
    # scatter = plt.scatter(x_positions, y_positions, color='grey', alpha=0.8, s=3)
    scatter = plt.scatter(x_positions, y_positions, c=codes, alpha=0.8, s=3, cmap='tab10', edgecolors='none')
    handles, labels = scatter.legend_elements(num=len(unique_clusters))
    plt.legend(handles, unique_clusters, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

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