import scanpy as sc 
import numpy as np 
import pandas as pd 
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from pqdm.processes import pqdm
from velocyto.estimation import colDeltaCorpartial

from .layout import view_XYZeq_3D


def compare_gex(adata, annot, goi, n_neighbors=15, n_pcs=20, show_paga=True, seed=123, figsize=(10,5)):
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.diffmap(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_diffmap')
        sc.tl.paga(adata, groups=annot)

        if show_paga:
            sc.pl.paga(adata)
        
        sc.tl.draw_graph(adata, init_pos='paga', random_state=seed)
        # sc.pl.draw_graph(adata, color=annot, legend_loc='on data', show=False)
        sc.pl.draw_graph(adata, color=[goi, annot],
                 layer="imputed_count", use_raw=False, cmap="viridis")

def plot_quiver(grid_points, vector_field, background=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if background is not None:
        ax.scatter(background['X'], background['Y'], c=background['annot'], alpha=0.3)

    magnitudes = np.linalg.norm(vector_field, axis=1)
    indices = magnitudes > 0
    grid_points = grid_points[indices]
    vector_field = vector_field[indices]

    ax.quiver(
        grid_points[:, 0], grid_points[:, 1],   
        vector_field[:, 0], vector_field[:, 1], 
        angles='xy', scale_units='xy', scale=1, 
        headwidth=3, headlength=3, headaxislength=3, width=0.002
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Estimated Transition Visualization')
    ax.set_axis_off()

    if ax is not None:
        return ax
    
    plt.show()

def plot_vectorfield(grid_points, vector_field, background=None):
    x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
    u, v, w = vector_field[..., 0].flatten(), vector_field[..., 1].flatten(), vector_field[..., 2].flatten()

    fig = go.Figure(data=go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        # colorscale='Viridis',
        sizemode="absolute",
        sizeref=1,
        showscale=False
    ))

    if background is not None:
        scatter_fig = px.scatter_3d(background, x='X', y='Y', z='Z', color='annot')
        scatter_fig.update_traces(marker=dict(size=2), line=dict(width=2, color='black'))

        for trace in scatter_fig.data:
            fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title='3D Estimated Transition Visualization'
    )
    fig.show()

def compute_probability(i, d_i, gene_mtx, indices, n_cells, T):
    exp_corr_sum = 0
    row_probs = np.zeros(n_cells)
    
    for j in indices[i]:
        r_ij = gene_mtx[i] - gene_mtx[j]
        corr, _ = pearsonr(r_ij, d_i)
        if np.isnan(corr):
            corr = 1
        exp_corr = np.exp(corr / T)
        exp_corr_sum += exp_corr
        row_probs[j] = exp_corr

    if exp_corr_sum != 0:
        row_probs /= exp_corr_sum

    return np.array(row_probs)

## CellOracle uses adapted Velocyto code
## This function is coded exactly as described in CellOracle paper
def estimate_transition_probabilities(adata, delta_X, n_neighbors=200, T=0.05, n_jobs=1):
    n_cells, n_genes = adata.shape
    delta_X = np.array(delta_X)
    gene_mtx = adata.layers['imputed_count']
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nn.fit(gene_mtx)
    _, indices = nn.kneighbors(gene_mtx)

    P = np.zeros((n_cells, n_cells))
    P[indices] = 1

    corr = colDeltaCorpartial(
        np.ascontiguousarray(gene_mtx.T), 
        np.ascontiguousarray(delta_X.T), 
        indices, threads=n_jobs)
    corr = np.nan_to_num(corr, nan=1)
    np.fill_diagonal(corr, 0)

    P *= np.exp(corr / T)   # naive
    P /= P.sum(1)[:, None]

    # args = [[i, delta_X[i], gene_mtx, indices, n_cells, T] for i in range(n_cells)]
    # results = pqdm(
    #     args,
    #     compute_probability,
    #     n_jobs=n_jobs,
    #     argument_type='args',
    #     tqdm_class=tqdm,
    #     desc='Estimating cell transition probabilities',
    # )

    # for i, row_probs in enumerate(results):
    #     P[i] = np.array(row_probs)

    return P

def project_probabilities(P, embedding):
    V_diff = embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :]
    V_simulated = np.einsum('ij,ijk->ik', P, V_diff)
    return V_simulated

def estimate_transitions_2D(adata, delta_X, embedding, annot=None, vector_scale=0.1, grid_size=20, n_jobs=1):

    if 'transition_probabilities' not in adata.uns.keys():
        P = estimate_transition_probabilities(adata, delta_X, n_jobs=n_jobs)
        adata.uns['transition_probabilities'] = P
    else:
        P = adata.uns['transition_probabilities']
    
    V_simulated = project_probabilities(P, embedding)

    L = grid_size
    grid_x = np.linspace(np.min(embedding[:, 0]), np.max(embedding[:, 0]), L)
    grid_y = np.linspace(np.min(embedding[:, 1]), np.max(embedding[:, 1]), L)

    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
    vector_field = np.zeros((L, L, 2))

    for idx, grid_point in tqdm(enumerate(grid_points), desc='Computing vectors', total=len(grid_points)):
        vector_distance = np.linalg.norm(embedding - grid_point, axis=1)
        total_weight = 0
        smoothed_vector = np.zeros(2)

        if np.min(vector_distance) < 0.1:
            
            # Sum weighted vectors within neighborhood
            for i in range(adata.n_obs):
                weight = gaussian_kernel(grid_point, embedding[i])
                smoothed_vector += weight * V_simulated[i]
                total_weight += weight
            
            if total_weight > 0:
                smoothed_vector /= total_weight
            
        grid_idx_x, grid_idx_y = np.unravel_index(idx, (L, L))
        vector_field[grid_idx_x, grid_idx_y] = smoothed_vector
    
    vector_field = vector_field.reshape(-1, 2)
    
    adata.uns['grid_points'] = grid_points
    adata.uns['vector_field'] = vector_field

    vector_field *= vector_scale
    if annot is None:
        background = None
    else:
        background = {
            'X': adata.obsm['spatial'][:, 0], 
            'Y': adata.obsm['spatial'][:, 1], 
            'annot': list(adata.obs[annot])
        }
    plot_quiver(grid_points, vector_field, background=background)

def estimate_transitions_3D(adata, delta_X, embedding, annot=None, vector_scale=0.1, grid_scale=1, n_jobs=1):
    if 'transition_probabilities' not in adata.uns.keys():
        P = estimate_transition_probabilities(adata, delta_X, n_jobs=n_jobs)
        adata.uns['transition_probabilities'] = P
    else:
        P = adata.uns['transition_probabilities']

    V_simulated = project_probabilities(P, embedding)
    
    get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, int(max_val - min_val + 1) * grid_scale)
    
    grid_x = get_grid_points(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    grid_y = get_grid_points(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    grid_z = get_grid_points(np.min(embedding[:, 2]), np.max(embedding[:, 2]))

    grid_points = np.array(np.meshgrid(grid_x, grid_y, grid_z)).T.reshape(-1, 3)
    size_x, size_y, size_z = len(grid_x), len(grid_y), len(grid_z)
    vector_field = np.zeros((size_x, size_y, size_z, 3))

    for idx, grid_point in tqdm(enumerate(grid_points), desc='Computing vectors', total=len(grid_points)):
        vector_distance = np.linalg.norm(embedding - grid_point, axis=1)
        total_weight = 0
        smoothed_vector = np.zeros(3)

        if np.min(vector_distance) < 0.1:

            # Sum weighted vectors within neighborhood
            for i in range(adata.n_obs):
                weight = gaussian_kernel(grid_point, embedding[i])
                smoothed_vector += weight * V_simulated[i]
                total_weight += weight
            
            if total_weight > 0:
                smoothed_vector /= total_weight
        
        grid_idx_x, grid_idx_y, grid_idx_z = np.unravel_index(idx, (size_x, size_y, size_z))
        vector_field[grid_idx_x, grid_idx_y, grid_idx_z] = smoothed_vector
    
    adata.uns['grid_points'] = grid_points
    adata.uns['vector_field'] = vector_field
    vector_field = adata.uns['vector_field'] * 0.01 / vector_scale
    
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


def gaussian_kernel(v0, v1, sigma=1.0):
        return np.exp(-np.linalg.norm(v0 - v1) ** 2 / (2 * sigma ** 2))