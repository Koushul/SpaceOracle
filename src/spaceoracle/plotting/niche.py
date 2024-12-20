import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from spaceoracle.models.parallel_estimators import received_ligands

import math
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import griddata


def get_modulator_betas(so_obj, goi, save_dir=None, use_simulated=False, clusters=[], blur=False):
    if so_obj.beta_dict is None:
        so_obj.beta_dict = so_obj._get_spatial_betas_dict() 
        
    beta_dict = so_obj.beta_dict.data

    if len(clusters) > 0:
        cell_idxs = np.where(so_obj.adata.obs[so_obj.annot_labels].isin(clusters))[0]
    else:
        cell_idxs = np.arange(len(so_obj.adata.obs))

    if use_simulated:
        gene_mtx = so_obj.adata.layers['simulated_count']
    else:
        gene_mtx = so_obj.adata.layers['imputed_count']

    gex_df = pd.DataFrame(gene_mtx, index=so_obj.adata.obs_names, columns=so_obj.adata.var_names)

    weighted_ligands = received_ligands(
        xy=so_obj.adata.obsm['spatial'], 
        lig_df=gex_df[list(so_obj.ligands)],
        radius=so_obj.radius
    )

    bois = []
    for gene, betaoutput in tqdm(beta_dict.items(), total=len(beta_dict), desc='Ligand interactions'):
        betas_df= so_obj._combine_gene_wbetas(gene, weighted_ligands, gex_df, betaoutput)        
        if f'beta_{goi}' in betas_df.columns:
            bois.append(betas_df[f'beta_{goi}'].rename(f'{gene}_beta_{goi}'))
    
    if len(bois) == 0:
        print(f'{goi} is not a modulator of any gene')
        return None
    
    df = pd.concat(bois, axis=1)

    beta_mean = df.mean(axis = 1) # average across all genes with beta_goi
    x = so_obj.adata.obsm['spatial'][:, 0][cell_idxs]
    y = so_obj.adata.obsm['spatial'][:, 1][cell_idxs]
    beta_mean = beta_mean.iloc[cell_idxs].to_numpy()
    
    if blur:
        plot_blur(x, y, beta_mean)
    else:
        plt.scatter(x, y, c=beta_mean, cmap='viridis', s = 0.5)
    plt.colorbar()
    plt.title(f'beta_{goi}')

    if save_dir:
        df.to_csv(os.path.join(save_dir, f'beta_{goi}_all.csv'))
        plt.savefig(os.path.join(save_dir, f'{goi}_heatmap.png'))
    
    plt.show()
    return df

def plot_blur(x, y, z, resolution=None):

    # Avoid plotting cells that are scattered far out
    coords = pd.DataFrame({'x': x, 'y': y})
    coords_count = coords.groupby(['x', 'y']).size()

    # Filter for coordinates that occur more than 3 times
    valid_coords = coords_count[coords_count > 3].index

    # Mask to keep only rows with valid coordinates
    mask = coords.apply(tuple, axis=1).isin(valid_coords)

    # Filter x, y, and beta_mean
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if resolution is None:
        resolution = max(x.max()-x.min(), y.max()-y.min()) * 3

    from scipy.interpolate import griddata

    def combine_stacked_points(x, y, z, agg_func=np.mean):
        coords = np.stack((x, y), axis=1)
        unique_coords, indices = np.unique(coords, axis=0, return_inverse=True)
        
        # Initialize aggregated z array
        aggregated_z = np.zeros(len(unique_coords))
        
        # Aggregate z values for each unique coordinate
        for idx in range(len(unique_coords)):
            aggregated_z[idx] = agg_func(z[indices == idx])
        
        return unique_coords, aggregated_z
    
    coords, z = combine_stacked_points(x, y, z)

    # Define grid for interpolation
    xi = np.linspace(x.min(), x.max(), resolution) 
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate `aggregated_z` onto the grid
    zi = griddata(coords, z, (xi, yi), method='linear')

    # Plot the interpolated data
    plt.imshow(
        zi, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis'
    )
    plt.scatter(x, y, c=z, cmap='viridis', s=0.5, edgecolor='none', alpha=0.9)



def show_beta_neighborhoods(so, goi, betas=None, annot=None, clusters=None, score_thresh=0.5, seed=1334, savepath=False):
    adata = so.adata
    if annot is None:
        annot = so.annot
    beta_dict = so.beta_dict
    if betas is None:
        betas = beta_dict.data[goi].iloc[:, :-4].values
    betas = np.array(betas)
    # cell_types = beta_dict.data[goi][annot]
    cell_types = adata.obs[annot].values
    if clusters is None:
        clusters = np.unique(cell_types)

    labels = np.full(len(betas), -1, dtype=int)
    range_n_clusters = range(2, 5)  # Range of clusters to try

    for cell_type in clusters:

        subset_idxs = np.where(cell_types == cell_type)[0]
        subset = betas[subset_idxs]

        best_score = score_thresh
        best_n_clusters = 1

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            cluster_labels = kmeans.fit_predict(subset)
            if len(set(cluster_labels)) > 1: 
                score = silhouette_score(subset, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

        print(cell_type, best_score)

        best_kmeans = KMeans(n_clusters=best_n_clusters, random_state=seed)
        best_labels = best_kmeans.fit_predict(subset)

        labels[subset_idxs] = best_labels + np.max(labels) + 1

    rows, cols = get_grid_layout(len(np.unique(labels)), preferred_cols=None)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()

    for i in np.unique(labels):
        cluster_mask = labels == i
        celltype = cell_types[cluster_mask][0]
        
        axes[i].scatter(
            adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
            c='lightgray', s=3, edgecolors='black', linewidth=0.1
        )
        
        axes[i].scatter(
            adata.obsm['spatial'][cluster_mask, 0], adata.obsm['spatial'][cluster_mask, 1],
            c='blue', s=3, edgecolors='black', linewidth=0.1
        )
        
        axes[i].set_title(f'Cluster {i} ({celltype})')
        axes[i].set_xticks([])  
        axes[i].set_yticks([])  
    
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()

    return labels


def get_grid_layout(n_items, preferred_cols=None):
    if preferred_cols:
        n_cols = min(preferred_cols, n_items)
    else:
        n_cols = int(math.ceil(math.sqrt(n_items)))  # Aim for a square-ish layout
        
    n_rows = int(math.ceil(n_items / n_cols))
    return n_rows, n_cols