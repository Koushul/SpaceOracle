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
import json
from stqdm import stqdm


warnings.filterwarnings("ignore", category=FutureWarning)

import spaceoracle
# from spaceoracle.tools.network import DayThreeRegulatoryNetwork
# from spaceoracle.models.estimators import ViTEstimatorV2, device

st.title(f'SpaceOracle Debugger Tool 🛠️️')


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
    adata_train = sc.read_h5ad('./notebooks/.cache/adata_train.h5ad')
    return adata_train  

    

adata_train = load_data()
clean_up_adata(adata_train)


st.write(adata_train)



# available_genes = filter(None, map(
#     spaceoracle.oracles.OracleQueue.extract_gene_name, glob.glob('./notebooks/models/*.pkl')))

# st.multiselect('Select a gene', available_genes)

# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['figure.dpi'] = 100
# sc.pl.embedding(adata_train, color=['rctd_celltypes', gene], layer='imputed_count', 
#                 basis="spatial", s=85, show=False, 
#                 edgecolor='black', linewidth=0.35, frameon=False)


# fig = plt.gcf()
# st.pyplot(fig)


# model_dict = spaceoracle.SpaceOracle.load_estimator(gene, save_dir='./notebooks/models')
# model = model_dict['model'].to(device)

# with torch.no_grad():
#     betas = model.forward(
#         torch.from_numpy(adata_train.obsm['spatial_maps'][:, ...]).float().to(device),
#         torch.from_numpy(np.array(adata_train.obs['rctd_cluster'])[:, ...]).long().to(device)
#     )

# betas = betas.cpu().numpy()

# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['figure.dpi'] = 100

# def plot_pair(indices):
#     f, axs = plt.subplots(1, 4, figsize=(20, 8), dpi=140, sharex=True, sharey=True)
#     axs = axs.flatten()

#     scatter_plots = []

#     for i, ax in zip(indices, axs): 
#         scatter = sns.scatterplot(x=adata_train.obsm['spatial'][:, 0], y=adata_train.obsm['spatial'][:, 1], 
#                     s=20, c=betas[:, i+1], cmap='rainbow', edgecolor='black', linewidth=0.35, 
#                     ax=ax
#         )
#         scatter_plots.append(scatter)

#     beta_means = list(betas.mean(0))
#     for ix, ax in zip(indices, axs):
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.set_title(f'{model_dict["regulators"][ix]}\n'+ r'$\mu$' + f'={beta_means[ix+1]:.3f}')
#         ax.set_xticks([])
#         ax.set_yticks([])
        
        
#     plt.tight_layout()
#     f.subplots_adjust(bottom=0.15)

#     # Add a colorbar
#     cbar_ax = f.add_axes([0.1, 0.05, 0.8, 0.02])
#     colorbar = f.colorbar(
#         scatter_plots[0].collections[0], cax=cbar_ax, orientation='horizontal')


#     plt.suptitle(f'Regulatory impact of \ntranscription factors on {gene} ', fontsize=18)
#     plt.subplots_adjust(top=0.825)

#     st.pyplot(f)

# for i in range(0, 8, 4):
#     plot_pair([i, i+1, i+2, i+3])


# df = pd.DataFrame(betas, columns=['intercept']+model_dict['regulators'])

# grn = DayThreeRegulatoryNetwork()

# tf = st.selectbox('Select a transcription factor', model_dict['regulators'], index=1)

# plt.rcParams['figure.figsize'] = (1, 1)
# plt.rcParams['figure.dpi'] = 40
# fig, ax = plt.subplots(figsize=(5, 2.5), dpi=40)

# for celltype in adata_train.obs['rctd_celltypes'].unique():
#     sns.kdeplot(
#         df[tf].values[adata_train.obs['rctd_celltypes'] == celltype], 
#         ax=ax, shade=True, label=celltype)

# ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}"))

# ax.set_ylabel("Density")
# ax.set_title(f"Distribution of {tf} coefficients by cluster")

# # Add legend
# # ax.legend(["Cluster 0", "Cluster 1", "Cluster 2"])

# # Remove top and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Adjust layout to prevent cutting off labels
# plt.tight_layout()
# plt.legend(loc='upper right', ncol=1, fontsize=6)

# st.pyplot(fig)

# alpha = 0.05
# values = []
# for k, link_data in grn.links_day3_1.items():
#     v = link_data.query(f'target == "{gene}" and source == "{tf}" and p < {alpha}')['coef_mean'].values
#     values.append((k, v))


# st.write(values)



import spaceoracle
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create SpaceOracle object
@st.cache_resource
def load_spaceoracle():
    so = spaceoracle.SpaceOracle(
        adata=adata_train,
        annot='rctd_cluster', 
        max_epochs=5, 
        learning_rate=4e-4, 
        spatial_dim=64,
        batch_size=256,
        init_betas='co',
        rotate_maps=True,
        cluster_grn=True,
        regularize=True,
    )
    
    # Load pre-computed beta_dict and coef_matrix
    with open('./notebooks/beta_dict.pickle', 'rb') as handle:
        so.beta_dict = pickle.load(handle)
    so.coef_matrix = pd.read_csv('./notebooks/coef_matrix.csv', index_col=0)
    
    return so

so = load_spaceoracle()


@st.cache_data(persist='disk', show_spinner=False)
def perturb_with_progress(target, n_propagation, ):
    gene_mtx = so.adata.to_df(layer='imputed_count').values
    target_index = so.gene2index[target]  
    simulation_input = gene_mtx.copy()
    simulation_input[:, target_index] = 0  # ko target gene
    delta_input = simulation_input - gene_mtx  # get delta X
    delta_simulated = delta_input.copy() 

    # if so.beta_dict is None:
    #     so.beta_dict = so._get_spatial_betas_dict()  # compute betas for all genes for all cells
    
    for n in range(n_propagation):
        _simulated = np.array([
            so._perturb_single_cell(delta_simulated, i, so.beta_dict) 
            for i in stqdm(range(so.adata.n_obs), desc=f"Cells (Iteration {n+1})")
        ])
        delta_simulated = np.array(_simulated)
        delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

        gem_tmp = gene_mtx + delta_simulated
        gem_tmp[gem_tmp < 0] = 0
        delta_simulated = gem_tmp - gene_mtx

    gem_simulated = gene_mtx + delta_simulated

    return gem_simulated


# Add a selectbox for target gene selection
target_gene = st.selectbox('Select a target gene for perturbation', adata_train.var_names.tolist(), index=0)

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 100
sc.pl.embedding(adata_train, color=['rctd_celltypes', target_gene], layer='imputed_count', 
                basis="spatial", s=85, show=False, 
                edgecolor='black', linewidth=0.35, frameon=False)


regulators = so.grn.get_regulators(adata_train, target_gene)
targets = so.grn.get_targets(adata_train, target_gene)

fig = plt.gcf()
st.pyplot(fig)

with st.expander(f'{target_gene} is regulated by {len(regulators)} TFs'):
    st.write(regulators)


with st.expander(f'{target_gene} regulates {len(targets)} genes'):
    st.write(targets)


cluster = st.selectbox('Select a cluster', adata_train.obs['rctd_celltypes'].unique())
genes = st.multiselect('Select a gene', adata_train.var_names.tolist())


    

# Add a button to trigger the perturbation
if st.button('Run Perturbation'):

    with st.spinner('Running CellOracle perturbation...'):
        # Perform perturbation using CellOracle
        perturbed_matrix_co = so.perturb_via_celloracle(
            gene_mtx=adata_train.to_df(layer='imputed_count'),
            target=target_gene, n_propagation=1
        )

    # Perform perturbation using SpaceOracle
    with st.spinner('Running SpaceOracle perturbation...'):
        

        perturbed_matrix_so = perturb_with_progress(
            target=target_gene,
            n_propagation=1
        )



    # Calculate delta matrices
    delta_so = adata_train.to_df(layer='imputed_count') - perturbed_matrix_so
    delta_co = adata_train.to_df(layer='imputed_count') - perturbed_matrix_co


    adata_train.layers['perturbed_so'] = perturbed_matrix_so
    adata_train.layers['perturbed_co'] = perturbed_matrix_co
    adata_train.layers['perturbed_so_delta'] = adata_train.to_df(layer='imputed_count') - perturbed_matrix_so
    adata_train.layers['perturbed_co_delta'] = adata_train.to_df(layer='imputed_count') - perturbed_matrix_co
    

    st.write(so.adata[so.adata.obs['rctd_celltypes'] == cluster])

    for gene in genes:
        fig = plt.subplots(figsize=(10, 6))

        sns.kdeplot(adata_train.to_df(
            layer='perturbed_co')[so.adata.obs['rctd_celltypes'] == cluster][gene].values, 
            label='CellOracle', color='red', shade=True, linewidth=1)

        sns.kdeplot(adata_train.to_df(
            layer='imputed_count')[so.adata.obs['rctd_celltypes'] == cluster][gene].values, 
            label='imputed_count', color='green', shade=True, linewidth=1)
        
        sns.kdeplot(adata_train.to_df(
            layer='perturbed_so')[so.adata.obs['rctd_celltypes'] == cluster][gene].values, 
            label='SpaceOracle', color='blue', shade=True, linewidth=1)

        plt.legend()
        plt.title(f'Distribution of {gene} expression after KO of {target_gene} in {cluster}')
        plt.xlabel('Expression level')
        plt.ylabel('Density')
        
        st.pyplot(plt.gcf())

