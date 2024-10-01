import os
os.environ["OMP_NESTED"] = "FALSE"
import anndata
import scanpy as sc
import pandas as pd 
import copy


immune_modules = [18, 19, 33, 48, 58, 70, 74, 76]
immgen_dict = {
    'B-cell': [33, 76],
    'T-cell': [18],
    'Cd4 T-cell': [70],
    'NK': [19],
    'Myeloid': [58, 74],
    'Dendritic': [48],
}

def get_markers(adata, nsearch=500):
    gene_df = pd.read_excel('../data/immgen/gene_assignment.xls')
    sc.pp.highly_variable_genes(adata, n_top_genes=nsearch)
    var_genes = adata.var[adata.var['highly_variable'] == True].index
    gene_df = gene_df[gene_df.Gene.isin(var_genes)]

    marker_genes_dict = {}
    for label, c_modules in immgen_dict.items():
        cluster_df = gene_df[gene_df['Coarse module'].isin(c_modules)]
        marker_genes_dict[label] = list(cluster_df['Gene'])
    
    return marker_genes_dict


def filter_clusters(adata, c=None):
    if c is None:
        c = []
    
    c = np.array(c).astype(str)
    mask = ~adata.obs['rctd_cluster'].astype(str).isin(c)
    filtered_adata = adata[mask].copy()
    
    return filtered_adata

def get_immune_genes(mouse=True):
    gene_df = pd.read_excel('../data/immgen/gene_assignment.xls')
    genes = gene_df[gene_df['Coarse module'].isin(immune_modules)]['Gene'].values
    genes = np.unique(genes)

    if mouse:
        genes = [gene.capitalize() if isinstance(gene, str) else gene for gene in genes]

    return genes

def process_adata(
        adata_train, 
        n_top_genes=5500, min_cells=10, min_counts=100, 
        include_genes=[],
        mouse=True
    ):

    if mouse: 
        mito = 'mt'
    else:
        mito = 'MT'

    adata_train = adata_train.copy()
    adata_train.var_names_make_unique()
    adata_train.var[mito] = adata_train.var_names.str.startswith(f"{mito}-")
    sc.pp.calculate_qc_metrics(adata_train, qc_vars=[mito], inplace=True)
    sc.pp.filter_cells(adata_train, min_counts=min_counts)
    adata_train = adata_train[adata_train.obs[f"pct_counts_{mito}"] < 20].copy()
    adata_train = adata_train[:, ~adata_train.var[mito]]
    sc.pp.filter_genes(adata_train, min_cells=min_cells)

    adata_train.layers["raw_count"] = adata_train.X

    sc.pp.normalize_total(adata_train, inplace=True)
    sc.pp.log1p(adata_train)
    sc.pp.highly_variable_genes(
        adata_train, flavor="seurat", n_top_genes=n_top_genes)
    
    mask = adata_train.var['highly_variable'] | adata_train.var_names.isin(include_genes)

    adata_train = adata_train[:, mask]
    sc.pp.normalize_per_cell(adata_train)


def get_imputed(adata_train, spatial_dim, annot):
    from spaceoracle import SpaceOracle

    adata_train.layers["normalized_count"] = adata_train.to_df().values

    SpaceOracle.imbue_adata_with_space(adata_train, spatial_dim=spatial_dim, annot=annot, in_place=True)
    pcs, _ = SpaceOracle.perform_PCA(adata_train)
    SpaceOracle.knn_imputation(adata_train, pcs)

    return adata_train


# backwards functionality for visium data

spatial_dim = 64
adata_train = sc.read_h5ad('../data/slideseq/day3_1.h5ad')
adata_test = sc.read_h5ad('../data/slideseq/day3_2.h5ad')

immune_genes = get_immune_genes(mouse=True)
adata_train = process_adata(adata_train, include_genes=immune_genes)
adata_test = process_adata(adata_train, include_genes=immune_genes)

SpaceOracle.imbue_adata_with_space(adata_train, spatial_dim=spatial_dim, in_place=True)
pcs = SpaceOracle.perform_PCA(adata_train)
SpaceOracle.knn_imputation(adata_train, pcs)

SpaceOracle.imbue_adata_with_space(adata_test, spatial_dim=spatial_dim, in_place=True)
pcs = SpaceOracle.perform_PCA(adata_test)
SpaceOracle.knn_imputation(adata_test, pcs)