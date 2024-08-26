import streamlit as st
import sys
sys.path.append('src')
import torch
import scipy.sparse as sp
import glob
import warnings
import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import glob
import anndata
import scanpy as sc
import numpy as np
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning)

import spaceoracle
from spaceoracle.tools.network import DayThreeRegulatoryNetwork
from spaceoracle.models.estimators import ViTEstimatorV2, device

st.title(f'SpaceOracle')

import json

with open('./data/celltype_assign.json', 'r') as f:
    celltype_assign = json.load(f)


def clean_up_adata(adata):
    fields_to_keep = ['cluster', 'rctd_cluster', 'rctd_celltypes']
    current_obs_fields = adata.obs.columns.tolist()
    excess_obs_fields = [field for field in current_obs_fields if field not in fields_to_keep]
    for field in excess_obs_fields:
        del adata.obs[field]
    
    current_var_fields = adata.var.columns.tolist()
    excess_var_fields = [field for field in current_var_fields 
        if field not in []]
    for field in excess_var_fields:
        del adata.var[field]

    del adata.uns


@st.cache_data
def load_data():

    # n_top_genes = 4000
    # min_cells = 10
    # min_counts = 350

    # adata_train = anndata.read_h5ad('./data/slideseq/day3_1.h5ad')
    # adata_test = anndata.read_h5ad('./data/slideseq/day3_2.h5ad')

    # adata_train.var_names_make_unique()
    # adata_train.var["mt"] = adata_train.var_names.str.startswith("mt-")
    # sc.pp.calculate_qc_metrics(adata_train, qc_vars=["mt"], inplace=True)
    # sc.pp.filter_cells(adata_train, min_counts=min_counts)
    # adata_train = adata_train[adata_train.obs["pct_counts_mt"] < 20].copy()
    # adata_train = adata_train[:, ~adata_train.var["mt"]]
    # sc.pp.filter_genes(adata_train, min_cells=min_cells)

    # adata_train.layers["raw_count"] = adata_train.X

    # sc.pp.normalize_total(adata_train, inplace=True)
    # sc.pp.log1p(adata_train)
    # sc.pp.highly_variable_genes(
    #     adata_train, flavor="seurat", n_top_genes=n_top_genes)

    # adata_train = adata_train[:, adata_train.var.highly_variable]


    # adata_test.var_names_make_unique()
    # adata_test.var["mt"] = adata_test.var_names.str.startswith("mt-")
    # sc.pp.calculate_qc_metrics(adata_test, qc_vars=["mt"], inplace=True)
    # sc.pp.filter_cells(adata_test, min_counts=min_counts)
    # adata_test = adata_test[adata_test.obs["pct_counts_mt"] < 20].copy()
    # adata_test = adata_test[:, ~adata_test.var["mt"]]
    # sc.pp.filter_genes(adata_test, min_cells=min_cells)

    # adata_test.layers["raw_count"] = adata_test.X

    # sc.pp.normalize_total(adata_test, inplace=True)
    # sc.pp.log1p(adata_test)
    # sc.pp.highly_variable_genes(
    #     adata_test, flavor="seurat", n_top_genes=n_top_genes)

    # adata_test = adata_test[:, adata_test.var.highly_variable]

    # adata_train = adata_train[:, adata_train.var_names.isin(np.intersect1d(adata_train.var_names, adata_test.var_names))]
    # adata_test = adata_test[:, adata_test.var_names.isin(np.intersect1d(adata_train.var_names, adata_test.var_names))]

    # del adata_test

    # adata_train.layers['imputed_count'] = adata_train.to_df().values

    # spaceoracle.SpaceOracle.imbue_adata_with_space(adata_train, spatial_dim=64, in_place=True)
    # pcs = spaceoracle.oracles.Oracle.perform_PCA(adata_train)
    # spaceoracle.oracles.Oracle.knn_imputation(adata_train, pcs)

    adata_train = sc.read_h5ad('./notebooks/.cache/adata_train.h5ad')


    adata_train.obs['rctd_celltypes'] = adata_train.obs['rctd_cluster'].astype(str).map(celltype_assign)

    return adata_train  

    

adata_train = load_data()
clean_up_adata(adata_train)


st.write(adata_train)

available_genes = filter(None, map(spaceoracle.oracles.OracleQueue.extract_gene_name, glob.glob('./notebooks/models/*.pkl')))
gene = st.selectbox('Select a gene', available_genes)

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 100
sc.pl.embedding(adata_train, color=['rctd_celltypes', gene], layer='imputed_count', 
                basis="spatial", s=85, show=False, 
                edgecolor='black', linewidth=0.35, frameon=False)


fig = plt.gcf()
st.pyplot(fig)


model_dict = spaceoracle.SpaceOracle.load_estimator(gene, save_dir='./notebooks/models')
model = model_dict['model'].to(device)

with torch.no_grad():
    betas = model.forward(
        torch.from_numpy(adata_train.obsm['spatial_maps'][:, ...]).float().to(device),
        torch.from_numpy(np.array(adata_train.obs['rctd_cluster'])[:, ...]).long().to(device)
    )

betas = betas.cpu().numpy()

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 100

def plot_pair(indices):
    f, axs = plt.subplots(1, 4, figsize=(20, 8), dpi=140, sharex=True, sharey=True)
    axs = axs.flatten()

    scatter_plots = []

    for i, ax in zip(indices, axs): 
        scatter = sns.scatterplot(x=adata_train.obsm['spatial'][:, 0], y=adata_train.obsm['spatial'][:, 1], 
                    s=20, c=betas[:, i+1], cmap='rainbow', edgecolor='black', linewidth=0.35, 
                    ax=ax
        )
        scatter_plots.append(scatter)

    beta_means = list(betas.mean(0))
    for ix, ax in zip(indices, axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(f'{model_dict["regulators"][ix]}\n'+ r'$\mu$' + f'={beta_means[ix+1]:.3f}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        
    plt.tight_layout()
    f.subplots_adjust(bottom=0.15)

    # Add a colorbar
    cbar_ax = f.add_axes([0.1, 0.05, 0.8, 0.02])
    colorbar = f.colorbar(
        scatter_plots[0].collections[0], cax=cbar_ax, orientation='horizontal')


    plt.suptitle(f'Regulatory impact of \ntranscription factors on {gene} ', fontsize=18)
    plt.subplots_adjust(top=0.825)

    st.pyplot(f)

for i in range(0, 8, 4):
    plot_pair([i, i+1, i+2, i+3])


df = pd.DataFrame(betas, columns=['intercept']+model_dict['regulators'])

grn = DayThreeRegulatoryNetwork()

tf = st.selectbox('Select a transcription factor', model_dict['regulators'], index=1)

plt.rcParams['figure.figsize'] = (1, 1)
plt.rcParams['figure.dpi'] = 40
fig, ax = plt.subplots(figsize=(5, 2.5), dpi=40)

for celltype in adata_train.obs['rctd_celltypes'].unique():
    sns.kdeplot(
        df[tf].values[adata_train.obs['rctd_celltypes'] == celltype], 
        ax=ax, shade=True, label=celltype)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}"))

ax.set_ylabel("Density")
ax.set_title(f"Distribution of {tf} coefficients by cluster")

# Add legend
# ax.legend(["Cluster 0", "Cluster 1", "Cluster 2"])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout to prevent cutting off labels
plt.tight_layout()
plt.legend(loc='upper right', ncol=1, fontsize=6)

st.pyplot(fig)

alpha = 0.05
values = []
for k, link_data in grn.links_day3_1.items():
    v = link_data.query(f'target == "{gene}" and source == "{tf}" and p < {alpha}')['coef_mean'].values
    values.append((k, v))


st.write(values)