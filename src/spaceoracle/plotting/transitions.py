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
from velocyto.estimation import colDeltaCorpartial, colDeltaCor

from .layout import view_spatial3D

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
        ax.scatter(background['X'], background['Y'], c=background['annot'], alpha=0.3, s=2)

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
    zrange = [np.min(z) - 0.1, np.max(z) + 0.1]
    # zrange = [np.min(z) - 0.6, np.max(z) + 0.6]

    fig = go.Figure(data=go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        colorscale='solar',
        showscale=True,
        colorbar=dict(title="Vector Intensity", len=0.5, x=1.1),
        sizemode="scaled",
        sizeref=1.5,
        anchor="tail",
        lighting=dict(diffuse=0.9, specular=0.1) 
    ))

    if background is not None:
        scatter_fig = px.scatter_3d(background, x='X', y='Y', z='Z', color='annot')
        scatter_fig.update_traces(marker=dict(size=1), line=dict(width=2, color='black'))

        for trace in scatter_fig.data:
            fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z', range=zrange),
        ),
        title='3D Estimated Transition Visualization',
        margin=dict(l=0, r=0, b=0, t=50), 
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)) 
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
def estimate_transition_probabilities(adata, delta_X, embedding=None, n_neighbors=200, random_neighbors=False, T=0.05, n_jobs=1):
    n_cells, n_genes = adata.shape
    delta_X = np.array(delta_X)
    gene_mtx = adata.layers['imputed_count']
    
    if random_neighbors:
        P = np.zeros((n_cells, n_cells))
        
        cells = np.arange(n_cells)
        for i in range(n_cells):
            indices = np.random.choice(np.delete(cells, i), size=n_neighbors, replace=False)
            P[i, indices] = 1
    elif n_neighbors:
        P = np.zeros((n_cells, n_cells))

        nn = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=n_jobs)
        nn.fit(embedding)
        _, indices = nn.kneighbors(embedding)

        rows = np.repeat(np.arange(n_cells), n_neighbors+1)
        cols = indices.flatten()
        P[rows, cols] = 1
    else:
        P = np.ones((n_cells, n_cells))
    
    np.fill_diagonal(P, 0)

    corr = colDeltaCorpartial(
        np.ascontiguousarray(gene_mtx.T), 
        np.ascontiguousarray(delta_X.T), 
        indices, threads=n_jobs
        )

    corr = np.nan_to_num(corr, nan=1)

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

def project_probabilities(P, embedding, normalize=True):
    if normalize: 
        embed_dim = embedding.shape[1]

        embedding_T = embedding.T # shape (m, n_cells)
        unitary_vectors = embedding_T[:, None, :] - embedding_T[:, :, None]  # shape (m, n_cells, n_cells)
        
        # Normalize the difference vectors (L2 norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            norms = np.linalg.norm(unitary_vectors, ord=2, axis=0)  # shape (n_cells, n_cells)
            unitary_vectors /= norms
            for m in range(embed_dim):
                np.fill_diagonal(unitary_vectors[m, ...], 0)   
        
        delta_embedding = (P * unitary_vectors).sum(2)  # shape (m, n_cells)
        delta_embedding = delta_embedding.T
        
        return delta_embedding
    
    else:
        embed_diffs = embedding[np.newaxis, :, :] - embedding[:, np.newaxis, :]
        
        # masked = embed_diffs * P[:, :, np.newaxis]
        # V_simulated = np.sum(masked, axis=1)
        V_simulated = np.einsum('ij,ijk->ik', P, embed_diffs)
        
        return V_simulated

def estimate_transitions_2D(adata, delta_X, embedding, annot=None, normalize=True, 
n_neighbors=200, vector_scale=0.1, grid_scale=1, n_jobs=1):

    P = estimate_transition_probabilities(adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=n_jobs)
    V_simulated = project_probabilities(P, embedding, normalize=normalize)

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
        smoothed_vector = np.zeros(2)

        # Get average vector within neighborhood
        indices = get_neighborhood(grid_point, embedding)
        if len(indices) <= 0:
            continue
        nbr_vector = np.mean(V_simulated[indices], axis=0)
            
        grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
        vector_field[grid_idx_x, grid_idx_y] = nbr_vector
    
    vector_field = vector_field.reshape(-1, 2)
    
    ### for testing, delete or save properly later
    adata.uns['grid_points'] = grid_points
    adata.uns['vector_field'] = vector_field
    adata.uns['P'] = P
    adata.uns['V_simulated'] = V_simulated
    ###

    vector_field *= vector_scale
    if annot is None:
        background = None
    else:
        background = {
            'X': embedding[:, 0], 
            'Y': embedding[:, 1], 
            'annot': list(adata.obs[annot])
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
        
        grid_idx_x, grid_idx_y, grid_idx_z = np.unravel_index(idx, (size_x, size_y, size_z))
        vector_field[grid_idx_x, grid_idx_y, grid_idx_z] = nbr_vector
    
    ### for testing, delete or save properly later
    adata.uns['grid_points'] = grid_points
    adata.uns['vector_field'] = vector_field
    adata.uns['P'] = P
    adata.uns['V_simulated'] = V_simulated
    ###

    vector_field = adata.uns['vector_field'] * vector_scale
    
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