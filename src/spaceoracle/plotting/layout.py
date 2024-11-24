import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

import numpy as np 
import pandas as pd 
import scanpy as sc 

def view_spatial2D(adata, annot, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)

    X = adata.obsm['spatial'][:, 0]
    Y = adata.obsm['spatial'][:, 1]

    categories = adata.obs[annot].astype('category')
    codes = categories.cat.codes

    scatter = plt.scatter(X, Y, c=codes, alpha=0.2, s=2, cmap='viridis')
    
    handles, labels = scatter.legend_elements(num=len(categories.cat.categories))
    category_labels = categories.cat.categories
    
    plt.legend(handles, category_labels, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.show()
    

def view_spatial3D(adata, annot, flat=False, show=True):
    df = pd.DataFrame({
        'X': adata.obsm['spatial'][:, 0],
        'Y': adata.obsm['spatial'][:, 1],
        'celltype': adata.obs[annot]
    })

    df = df.sort_values(by='celltype').reset_index(drop=False)
    Z = np.zeros(len(df))
    
    for ct, celltype in enumerate(df['celltype'].unique()):
        celltype_df = df[df['celltype'] == celltype]
        for i, (x, y) in enumerate(zip(celltype_df['X'], celltype_df['Y'])):
            if flat: 
                Z[celltype_df.index[i]] = ct
            else: 
                Z[celltype_df.index[i]] = (ct * 10) + np.random.choice(10)


    df['Z'] = Z

    df.set_index('index', inplace=True)
    df = df.reindex(adata.obs.index)
    adata.obsm['spatial_3D'] = df[['X','Y','Z']].values

    if not show:
        return df[['X','Y','Z']].values


    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='celltype')
    fig.update_traces(marker=dict(size=2), line=dict(width=2, color='black'))
    fig.show()


def compare_gex(adata, annot, goi, embedding='FR', n_neighbors=15, n_pcs=20, seed=123, savepath=False):

    assert embedding in ['FR', 'PCA', 'UMAP', 'spatial'], f'{embedding} is not a valid embedding choice'
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    if embedding == 'PCA':
        sc.pl.pca(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis', show=False)
    
    elif embedding == 'UMAP':
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis', show=False)

    elif embedding == 'spatial':
        x = adata.obsm['spatial'][:, 0]
        y = adata.obsm['spatial'][:, 1] * -1

        adata = adata.copy()
        adata.obsm['spatial'] = np.vstack([x, y]).T
        sc.pl.spatial(
            adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis', spot_size=50, show=False)

    elif embedding == 'FR': 

        sc.tl.diffmap(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_diffmap')
        sc.tl.paga(adata, groups=annot)
        sc.pl.paga(adata)

        sc.tl.draw_graph(adata, init_pos='paga', random_state=seed)
        sc.pl.draw_graph(adata, color=[goi, annot], layer="imputed_count", use_raw=False, cmap="viridis", show=False)

    if savepath: 
        plt.savefig(savepath, dpi=300) 

    plt.show()
    plt.close()



def plot_quiver(grid_points, vector_field, background=None, ax=None, savepath=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)  # Higher DPI and larger figure for better resolution

    # Plot cells in background if given
    if background is not None:
        cmap = plt.get_cmap('tab20')
        celltypes = np.unique(background['annot'])
        category_colors = {ct: cmap((i*2) + round(i / 2)) for i, ct in enumerate(celltypes)}
        colors = [category_colors[ct] for ct in background['annot']]

        scatter = ax.scatter(
            background['X'], 
            background['Y'], 
            c=colors, 
            alpha=0.5,  # Slightly higher opacity for better visibility
            s=10,  # Larger point size for better visibility
            edgecolor='black', 
            linewidth=0.2  # Thin black edge for contrast
        )
    
    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=ct)
        for ct, color in category_colors.items()
    ]
    ax.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Filter out zero-magnitude vectors
    magnitudes = np.linalg.norm(vector_field, axis=1)
    indices = magnitudes > 0
    grid_points = grid_points[indices]
    vector_field = vector_field[indices]

    # Plot quiver vectors 
    ax.quiver(
        grid_points[:, 0], grid_points[:, 1],   
        vector_field[:, 0], vector_field[:, 1], 
        angles='xy', scale_units='xy', scale=1, 
        headwidth=4, headlength=6, headaxislength=4,  # Larger arrowheads
        color='black',  # Solid black arrows for contrast
        width=0.003, alpha=0.8
    )

    # Set labels, title, and adjust visibility
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('2D Transition Visualization', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    
    if ax is not None:
        return ax

    plt.show()

def get_quiver_grid(embedding, grid_scale, V_simulated):  

    base_grid_size = 50 / np.mean(abs(np.diff(embedding)))
    grid_scale = grid_scale * base_grid_size

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
    return grid_points, vector_field