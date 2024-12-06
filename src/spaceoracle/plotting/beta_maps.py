import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Define markers and colormap list
markers = ['o', 'X', '<', '^', 'v', 'D', '>']
colormap_list = ['rainbow', 'cool', 'RdYlGn_r', 'spring_r', 'PuRd', 'Reds']
cmaps = dict(zip(range(20), cycle(colormap_list)))  # Wrap around for 20 indices


def plot_spatial(df, plot_for, target_gene=None, clusters=[0, 1, 2], with_expr=False, 
                 annot='rctd_clusters', annot_labels='rctd_celltypes', 
                 size=25, linewidth=0.5, alpha=1, edgecolor='black', 
                 dpi=100, figsize=(11, 9), fig=None, axes=None):
    
    # Create a mapping from clusters to cell types
    cell_map = dict(zip(df[annot], df[annot_labels]))

    # Initialize the figure and axes
    if fig is None or axes is None:
        fig, (ax, cax) = plt.subplots(1, 2, dpi=dpi, figsize=figsize, gridspec_kw={'width_ratios': [4, 0.3]})
    else:
        ax, cax = axes

    # Create normalization for each cluster
    norms = {}
    for i in clusters:
        cluster_data = df[df[annot] == i][plot_for]
        vmin = cluster_data.min()
        vmax = cluster_data.max()
        norms[i] = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot each cluster
    for i in clusters:
        betas_df = df[df[annot] == i]
        if with_expr:
            betas_df[plot_for] = betas_df[plot_for] * betas_df[plot_for.replace('beta_', '')]

        sns.scatterplot(
            data=betas_df,
            x='x', 
            y='y',
            hue=plot_for,
            palette=cmaps.get(i, 'viridis'),  # Default to 'viridis' if cmap is invalid
            s=size,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor=edgecolor,
            legend=False,
            style=annot_labels,
            markers=markers[:len(set(df[annot_labels]))],  # Adjust markers to match unique styles
            ax=ax
        )

    ax.axis('off')

    # Adjust colorbar layout
    cbar_width = 0.1  # Width of each colorbar
    cbar_height = 0.2  # Height of each colorbar
    spacing = 0.25 / len(clusters)  # Dynamic spacing based on the number of clusters
    for idx, i in enumerate(clusters):
        cmap_name = cmaps.get(i, 'viridis')
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norms[i])
        sm.set_array([])

        # Adjust placement for better layout
        cax_i = cax.inset_axes([0.2, 1 - (idx + 1) * cbar_height - idx * spacing, cbar_width, cbar_height])
        cbar = fig.colorbar(sm, cax=cax_i, orientation='vertical')
        cbar.ax.tick_params(labelsize=9)  # Reduce tick label size
        cbar.ax.set_title(f'{cell_map.get(i, "Unknown")}', fontsize=12, pad=5)  # Handle missing labels

    cax.axis('off')

    # Generate legend for cell types
    unique_styles = sorted(set(df[annot_labels]))
    style_handles = [
        plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=10, linestyle='None', alpha=1)
        for m in markers[:len(unique_styles)]  # Match markers to unique styles
    ]
    ax.legend(
        style_handles, unique_styles, ncol=1,
        title='Cell types', loc='lower left',  
        frameon=False
    )
    ax.set_title(f'{plot_for} > {target_gene}', fontsize=15)
    
    return ax
