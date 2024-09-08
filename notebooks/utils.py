import os
os.environ["OMP_NESTED"] = "FALSE"
import anndata
import scanpy as sc
import numpy as np
from spaceoracle import SpaceOracle


spatial_dim = 64
# genes_of_interest = ['Il2', 'Il2ra', 'Prdm1']


def filter_clusters(adata, c=None):
    if c is None:
        c = []
    
    c = np.array(c).astype(str)
    mask = ~adata.obs['rctd_cluster'].astype(str).isin(c)
    filtered_adata = adata[mask].copy()
    
    return filtered_adata


def process_adata(adata, n_top_genes=3000, min_cells=10, min_counts=200):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    adata = adata[:, ~adata.var["mt"]]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.layers["raw_count"] = adata.X

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes)

    adata.layers["normalized_count"] = adata.to_df().values

    # adata = adata[:, (adata.var.highly_variable | adata.var_names.isin(genes_of_interest))]
    return adata



# adata_train = anndata.read_h5ad('../data/slideseq/day3_1.h5ad')
# adata_test = anndata.read_h5ad('../data/slideseq/day3_2.h5ad')

# adata_train = process_adata(adata_train)
# adata_test = process_adata(adata_test)


# adata = adata[:, adata.var_names.isin(
#     np.intersect1d(adata.var_names, adata_test.var_names))]
# adata_test = adata_test[:, adata_test.var_names.isin(
#     np.intersect1d(adata.var_names, adata_test.var_names))]


# SpaceOracle.imbue_adata_with_space(adata, spatial_dim=spatial_dim, in_place=True)
# pcs = SpaceOracle.perform_PCA(adata)
# SpaceOracle.knn_imputation(adata, pcs)

# SpaceOracle.imbue_adata_with_space(adata_test, spatial_dim=spatial_dim, in_place=True)
# pcs = SpaceOracle.perform_PCA(adata_test)
# SpaceOracle.knn_imputation(adata_test, pcs)