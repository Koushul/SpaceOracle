import anndata
import scanpy as sc
import numpy as np
from spaceoracle import SpaceOracle

import os
os.environ["OMP_NESTED"] = "FALSE"
import anndata
import scanpy as sc
import numpy as np
from spaceoracle import SpaceOracle


n_top_genes = 5500
min_cells = 10
min_counts = 100
spatial_dim = 64

genes_of_interest = ['Il2', 'Il2ra', 'Prdm1']

adata_train = anndata.read_h5ad('../data/slideseq/day3_1.h5ad')
adata_test = anndata.read_h5ad('../data/slideseq/day3_2.h5ad')

adata_train.var_names_make_unique()
adata_train.var["mt"] = adata_train.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata_train, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata_train, min_counts=min_counts)
adata_train = adata_train[adata_train.obs["pct_counts_mt"] < 20].copy()
adata_train = adata_train[:, ~adata_train.var["mt"]]
sc.pp.filter_genes(adata_train, min_cells=min_cells)

adata_train.layers["raw_count"] = adata_train.X

sc.pp.normalize_total(adata_train, inplace=True)
sc.pp.log1p(adata_train)
sc.pp.highly_variable_genes(
    adata_train, flavor="seurat", n_top_genes=n_top_genes)

adata_train = adata_train[:, (adata_train.var.highly_variable | adata_train.var_names.isin(genes_of_interest))]


adata_test.var_names_make_unique()
adata_test.var["mt"] = adata_test.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata_test, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata_test, min_counts=min_counts)
adata_test = adata_test[adata_test.obs["pct_counts_mt"] < 20].copy()
adata_test = adata_test[:, ~adata_test.var["mt"]]
sc.pp.filter_genes(adata_test, min_cells=min_cells)

adata_test.layers["raw_count"] = adata_test.X

sc.pp.normalize_total(adata_test, inplace=True)
sc.pp.log1p(adata_test)
sc.pp.highly_variable_genes(
    adata_test, flavor="seurat", n_top_genes=n_top_genes)

adata_test = adata_test[:, (adata_test.var.highly_variable | adata_test.var_names.isin(genes_of_interest))]

adata_train = adata_train[:, adata_train.var_names.isin(
    np.intersect1d(adata_train.var_names, adata_test.var_names))]
adata_test = adata_test[:, adata_test.var_names.isin(
    np.intersect1d(adata_train.var_names, adata_test.var_names))]

adata_train.layers["normalized_count"] = adata_train.to_df().values
adata_test.layers["normalized_count"] = adata_test.to_df().values

SpaceOracle.imbue_adata_with_space(adata_train, spatial_dim=spatial_dim, in_place=True)
pcs = SpaceOracle.perform_PCA(adata_train)
SpaceOracle.knn_imputation(adata_train, pcs)

SpaceOracle.imbue_adata_with_space(adata_test, spatial_dim=spatial_dim, in_place=True)
pcs = SpaceOracle.perform_PCA(adata_test)
SpaceOracle.knn_imputation(adata_test, pcs)