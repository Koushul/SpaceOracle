print('init')
exit()


import sys
sys.path.append('../src')

from spaceoracle.models.estimators import ViTEstimatorV2

import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import seaborn as sns
import torch

adata_train = anndata.read_h5ad('../data/slideseq/day3_1.h5ad')
adata_test = anndata.read_h5ad('../data/slideseq/day3_2.h5ad')


n_top_genes = 4000
min_cells = 10
min_counts = 350

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

adata_train = adata_train[:, adata_train.var.highly_variable]

estimator = ViTEstimatorV2(adata_train, target_gene='Cd74')
print(estimator.regulators)

estimator.fit(
    annot='rctd_cluster',
    max_epochs=10,
    learning_rate=3e-4,
    spatial_dim=16,
    batch_size=32,
    init_betas='co',
    mode='train_test',
    rotate_maps=True,
    regularize=False,
    n_patches=16,
    n_heads=4,
    n_blocks=3,
    hidden_d=16
)

estimator.fit(
    annot='rctd_cluster',
    max_epochs=10,
    learning_rate=3e-4,
    spatial_dim=64,
    batch_size=32,
    init_betas='ols',
    mode='train_test',
    rotate_maps=True,
    regularize=False,
    n_patches=16,
    n_heads=4,
    n_blocks=3,
    hidden_d=16
)

estimator.fit(
    annot='rctd_cluster',
    max_epochs=10,
    learning_rate=3e-4,
    spatial_dim=64,
    batch_size=32,
    init_betas='ones',
    mode='train_test',
    rotate_maps=True,
    regularize=False,
    n_patches=16,
    n_heads=4,
    n_blocks=3,
    hidden_d=16
)