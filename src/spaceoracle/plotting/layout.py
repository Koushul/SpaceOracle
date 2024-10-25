import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns 

from collections import defaultdict
import numpy as np 
import pandas as pd 

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


def view_betas(so, regulator, target_gene, celltypes=None):
    markers = ['o', 'X', '<', '^', 'v', 'D', '>']
    cmaps = dict(zip(range(7), ['rainbow', 'cool', 'RdYlGn_r', 'spring_r', '', 'PuRd', 'Reds']))
    # cmap = 'rainbow'
    
    betadata = so.load_betadata(target_gene, so.save_dir)
    beta_columns = [i for i in betadata.columns if i[:5] == 'beta_' and '$' not in i]
    all_modulators = [i.replace('beta_', '') for i in beta_columns]

    df = betadata

    plot_for = regulator
    cell_map = dict(zip(df[so.annot], df[so.annot]))

    fig, (ax, cax) = plt.subplots(1, 2, dpi=80, figsize=(11, 9), gridspec_kw={'width_ratios': [4, 0.5]})

    if celltypes is None:
        celltypes = np.unique(df[so.annot])

    for i in celltypes:
        betas_df = df[['beta0']+['beta_'+i for i in all_modulators]][df[so.annot]==i]

        sns.scatterplot(
            data=betas_df.join(df[['x', 'y', so.annot]]),
            x='x', 
            y='y',
            hue=plot_for,
            palette=cmaps[i],
            s=25,
            alpha=1,
            linewidth=0.5,
            edgecolor='black',
            legend=False,
            style=so.annot,
            markers=markers,
            ax=ax
        )
    ax.axis('off')

    norm = None
    cbar_width = 0.15  # Width of each colorbar
    cbar_height = 0.8 / len(cmaps)  # Height of each colorbar
    for i, cmap_name in cmaps.items():
        if i not in [0, 1, 2]:
            continue
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax_i = cax.inset_axes([0.2, 0.95 - (i+1)*cbar_height*2.5, cbar_width, cbar_height*1.5])
        cbar = fig.colorbar(sm, cax=cax_i, orientation='vertical')
        cbar.ax.tick_params(labelsize=9)  # Reduce tick label size
        cbar.ax.set_title(f'{cell_map[i]}', fontsize=12, pad=8)  # Reduce title size and padding

    cax.set_ylabel(plot_for, fontsize=8)
    cax.axis('off')

    unique_styles = sorted(set(df['rctd_celltypes']))
    style_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', 
                    markersize=10, linestyle='None', alpha=1) 
                    for m in markers][:len(unique_styles)]
    ax.legend(style_handles, unique_styles, ncol=1,
        title='Cell types', loc='upper left', 
        frameon=False)

    ax.set_title(f'{plot_for} > {target_gene}', fontsize=15)
    plt.tight_layout()
    plt.show()