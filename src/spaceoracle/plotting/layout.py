import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go

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


def compare_gex(adata, annot, goi, embedding='FR', n_neighbors=15, n_pcs=20, seed=123):

    assert embedding in ['FR', 'PCA', 'UMAP', 'spatial'], f'{embedding} is not a valid embedding choice'
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    if embedding == 'PCA':
        sc.pl.pca(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis')
    
    elif embedding == 'UMAP':
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis')

    elif embedding == 'spatial':
        x = adata.obsm['spatial'][:, 0]
        y = adata.obsm['spatial'][:, 1] * -1

        adata = adata.copy()
        adata.obsm['spatial'] = np.vstack([x, y]).T
        sc.pl.spatial(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis', spot_size=50)

    elif embedding == 'FR': 

        sc.tl.diffmap(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_diffmap')
        sc.tl.paga(adata, groups=annot)
        sc.pl.paga(adata)

        sc.tl.draw_graph(adata, init_pos='paga', random_state=seed)
        sc.pl.draw_graph(adata, color=[goi, annot], layer="imputed_count", use_raw=False, cmap="viridis")



def plot_quiver(grid_points, vector_field, background=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    if background is not None:
        cmap = plt.get_cmap('tab20')
        celltypes = np.unique(background['annot'])
        category_colors = {ct: cmap(i / len(celltypes)) for i, ct in enumerate(celltypes)}
        colors = [category_colors[ct] for ct in background['annot']]

        ax.scatter(background['X'], background['Y'], c=colors, alpha=0.3, s=2)

    magnitudes = np.linalg.norm(vector_field, axis=1)
    indices = magnitudes > 0
    grid_points = grid_points[indices]
    vector_field = vector_field[indices]

    ax.quiver(
        grid_points[:, 0], grid_points[:, 1],   
        vector_field[:, 0], vector_field[:, 1], 
        angles='xy', scale_units='xy', scale=1, 
        headwidth=3, headlength=3, headaxislength=3,
        width=0.002, alpha=0.9
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
