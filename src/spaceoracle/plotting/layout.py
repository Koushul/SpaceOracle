import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns 
import os

from collections import defaultdict
import numpy as np 
import pandas as pd 
import scanpy as sc 
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from spaceoracle.models.parallel_estimators import received_ligands
from sklearn.cluster import KMeans, DBSCAN



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


def get_modulator_betas(so_obj, goi, save_dir=None, show=True):
    if so_obj.beta_dict is None:
        so_obj.beta_dict = so_obj._get_spatial_betas_dict() 
        
    beta_dict = so_obj.beta_dict
        
    gene_mtx = so_obj.adata.layers[so_obj.layer]
    gene_mtx = MinMaxScaler().fit_transform(gene_mtx)

    gex_df = pd.DataFrame(gene_mtx, index=so_obj.adata.obs_names, columns=so_obj.adata.var_names)

    weighted_ligands = received_ligands(
        xy=so_obj.adata.obsm['spatial'], 
        lig_df=gex_df[list(so_obj.ligands)],
        radius=so_obj.radius
    )

    bois = []
    for gene, betaoutput in tqdm(beta_dict.items(), total=len(beta_dict), desc='Ligand interactions'):
        gene, betas_df= so_obj._combine_gene_wbetas(gene, weighted_ligands, gex_df, betaoutput)
        if f'beta_{goi}' in betas_df.columns:
            bois.append(betas_df[f'beta_{goi}'].rename(f'{gene}_beta_{goi}'))
    df = pd.concat(bois, axis=1)

    if save_dir:
        df.to_csv(os.path.join(save_dir, f'beta_{goi}_all.csv'))

    if show:
        beta_mean = df.mean(axis = 1)
        plt.scatter(
            so_obj.adata.obsm['spatial'][:, 0], 
            so_obj.adata.obsm['spatial'][:, 1], 
            c=beta_mean, cmap='viridis', s=0.5
        )
        plt.colorbar()

    return df


def show_beta_neighborhoods(adata, betas, nneighborhoods=20, seed=1334, split_spatially=False):

    kmeans = KMeans(n_clusters=nneighborhoods, random_state=seed).fit(betas)
    labels = kmeans.labels_

    x_positions = adata.obsm['spatial'][:, 0]
    y_positions = adata.obsm['spatial'][:, 1]

    if split_spatially:
        new_labels = np.empty(len(x_positions), dtype=str)

        for cluster_id in np.unique(labels):

            cluster_points = np.vstack((
                x_positions[labels == cluster_id],
                y_positions[labels == cluster_id]
            )).T

            dbscan = DBSCAN(eps=10, min_samples=100)
            subcluster_labels = dbscan.fit_predict(cluster_points)

            subcluster_labels = [
                f"{cluster_id}_{label}" if label != -1 else f"{cluster_id}_noise"
                for label in subcluster_labels
            ]

            idxs = np.where(labels == cluster_id)
            new_labels[idxs] = subcluster_labels

        labels = np.array(new_labels)
        nneighborhoods = len(np.unique(labels)) 


    colors = plt.cm.tab20(np.linspace(0, 1, nneighborhoods))
    cmap = {label: colors[i] for i, label in enumerate(np.unique(labels))}

    fig, ax = plt.subplots(figsize=(6, 6))

    for label in np.unique(labels):
        group_cells = np.where(labels == label)[0]
        x = x_positions[group_cells]
        y = y_positions[group_cells]

        ax.scatter(x, y, color=cmap[label], alpha=0.8, s=3, label=f"Cluster {label}")

    ax.set_title("Neighborhoods from Betas")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return labels