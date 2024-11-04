import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def calculate_pca_embedding(adata, n_components=4):
    """Calculate PCA embedding of gene expression data"""
    pca = PCA(n_components=n_components)
    # Use imputed counts if available, otherwise normalized counts
    if 'imputed_count' in adata.layers:
        X = adata.to_df(layer='imputed_count')
    else:
        X = adata.to_df()
    return pca.fit_transform(X)


def plot_similarity_to_point(adata_train, embedding, point_idx, cluster, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    # Calculate cosine similarities
    similarities = cosine_similarity(embedding[point_idx:point_idx+1], embedding).flatten()
    
    # Get spatial coordinates
    spatial_coords = adata_train[adata_train.obs.rctd_cluster==cluster].obsm['spatial']
    
    # Create scatter plot
    scatter = ax.scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        c=similarities,
        cmap='viridis',
        s=50,
        alpha=1,
        linewidths=0.5,
        edgecolor='black'

    )
    
    # Highlight reference point
    ax.scatter(
        spatial_coords[point_idx, 0],
        spatial_coords[point_idx, 1],
        s=50,
        c='red',
        label='Reference point',
        linewidths=0.75,
        edgecolor='red'
    )


    top_5_indices = np.argsort(similarities)[:-1][-5:]
    
    # Add arrows and labels for top 5 similar points
    for i, idx in enumerate(top_5_indices):
        # Draw arrow from reference to similar point
        ax.annotate(
            f'#{i+1}',
            xy=(spatial_coords[idx, 0], spatial_coords[idx, 1]),
            xytext=(10, 10),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
        # Add larger point marker
        ax.scatter(
            spatial_coords[idx, 0],
            spatial_coords[idx, 1],
            color='yellow',
            s=100,
            alpha=0.5,
            zorder=2
        )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Transcriptional similarity to reference point')
    plt.colorbar(scatter, ax=ax, label='Cosine Similarity')
    ax.legend()
    
    return ax


def find_closest_point(array_2d, target_coord):
    distances = np.sqrt(np.sum((array_2d - target_coord)**2, axis=1))
    return np.argmin(distances)


def plot_similarity_chains(adata_train, annot, n=5, ax=None, cluster=0, figsize=(3, 3)):
    """
    Plot arrows showing the direction to each point's closest most similar neighbor.
    
    Parameters:
    -----------
    adata_train: AnnData
        Annotated data matrix containing spatial information.
    embedding: np.array
        PCA embedding of shape (n_cells, n_components).
    n: int 
        Number of points to plot chains for.
    ax: matplotlib axis, optional
        Axis to plot on.
    cluster: int
        Cluster ID to filter cells.
    figsize: tuple
        Size of the figure if `ax` is not provided.
    
    Returns:
    --------
    ax: matplotlib axis
        The axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    

    # Get spatial coordinates of points in the specified cluster
    adata_train = adata_train.copy()
    adata_train = adata_train[adata_train.obs[annot] == cluster]
    spatial_coords = adata_train.obsm['spatial']
    
    # Plot all points in gray
    ax.scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        color='gray',
        s=30,
        alpha=0.3
    )
    
    embedding = calculate_pca_embedding(adata_train, n_components=n)

    # Loop over the first `n` points
    for i in range(n):
        # Calculate cosine similarities to all points in the embedding
        similarities = cosine_similarity(embedding[i:i+1], embedding).flatten()
        similarities[i] = -np.inf  # Exclude self-similarity
        
        # Get indices of top 5 most similar points
        top_5_indices = np.argsort(similarities)[-5:]
        
        # Find the spatially closest point among top 5 similar points
        current_point = spatial_coords[i]
        top_5_coords = spatial_coords[top_5_indices]
        closest_idx = find_closest_point(top_5_coords, current_point)
        closest_similar_point = top_5_coords[closest_idx]
        
        # Draw an arrow from the current point to the closest similar point
        ax.annotate(
            '',
            xy=(closest_similar_point[0], closest_similar_point[1]),
            xytext=(current_point[0], current_point[1]),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6, lw=1.5)
        )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Arrows to closest transcriptionally similar points')
    
    return ax