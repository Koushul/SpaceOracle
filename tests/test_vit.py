import pytest
import numpy as np
import torch
import anndata as ad
import sys
import os
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from spaceoracle import SpaceOracle
from spaceoracle.models.estimators import ViTEstimatorV2, GeoCNNEstimatorV2
from spaceoracle.models.vit_blocks import ViT

@pytest.fixture
def mock_adata():
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame({
        'rctd_cluster': np.random.choice([0, 1, 2], size=n_obs),
    })
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars)])
    adata = ad.AnnData(X, obs=obs, var=var)
    adata.obsm['spatial'] = np.random.rand(n_obs, 2)
    return adata


def test_vit_forward_pass():
    batch_size = 10
    in_channels = 3
    spatial_dim = 64
    n_patches = 8
    n_blocks = 2
    hidden_d = 16
    n_heads = 2
    
    betas = torch.rand(5)  # Assuming 5 betas
    
    vit = ViT(betas, in_channels, spatial_dim, n_patches, n_blocks, hidden_d, n_heads)
    
    images = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    labels = torch.randint(0, 3, (batch_size,))  # Assuming 3 possible labels
    
    output = vit(images, labels)
    
    assert output.shape == (batch_size, len(betas))

def test_vit_attention_weights():
    batch_size = 10
    in_channels = 3
    spatial_dim = 64
    n_patches = 8
    n_blocks = 2
    hidden_d = 16
    n_heads = 2
    
    betas = torch.rand(5)  # Assuming 5 betas
    
    vit = ViT(betas, in_channels, spatial_dim, n_patches, n_blocks, hidden_d, n_heads)
    
    images = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    
    att_weights = vit.get_att_weights(images)
    
    assert len(att_weights) == n_blocks


def test_vit_with_real_data():
    adata_train = ad.read_h5ad('./data/slideseq/day3_1.h5ad')
    adata_test = ad.read_h5ad('./data/slideseq/day3_2.h5ad')

    n_top_genes = 4000
    min_cells = 10
    min_counts = 350
    spatial_dim = 64

    adata_train.var_names_make_unique()
    adata_train.var["mt"] = adata_train.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata_train, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata_train, min_counts=min_counts)
    adata_train = adata_train[adata_train.obs["pct_counts_mt"] < 20].copy()
    adata_train = adata_train[:, ~adata_train.var["mt"]].copy()
    sc.pp.filter_genes(adata_train, min_cells=min_cells)

    adata_train.layers["raw_count"] = adata_train.X

    sc.pp.normalize_total(adata_train, inplace=True)
    sc.pp.log1p(adata_train)
    sc.pp.highly_variable_genes(
        adata_train, flavor="seurat", n_top_genes=n_top_genes)

    adata_train = adata_train[:, adata_train.var.highly_variable]
    
    
    adata_test.var_names_make_unique()
    adata_test.var["mt"] = adata_test.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata_test, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata_test, min_counts=min_counts)
    adata_test = adata_test[adata_test.obs["pct_counts_mt"] < 20].copy()
    adata_test = adata_test[:, ~adata_test.var["mt"]].copy()
    sc.pp.filter_genes(adata_test, min_cells=min_cells)

    adata_test.layers["raw_count"] = adata_test.X

    sc.pp.normalize_total(adata_test, inplace=True)
    sc.pp.log1p(adata_test)
    sc.pp.highly_variable_genes(
        adata_test, flavor="seurat", n_top_genes=n_top_genes)

    adata_test = adata_test[:, adata_test.var.highly_variable]
    

    adata_train = adata_train[:, adata_train.var_names.isin(
        np.intersect1d(adata_train.var_names, adata_test.var_names))]
    adata_test = adata_test[:, adata_test.var_names.isin(
        np.intersect1d(adata_train.var_names, adata_test.var_names))]

    adata_train = adata_train.copy()
    adata_test = adata_test.copy()
    adata_train.layers["normalized_count"] = adata_train.to_df().values
    adata_test.layers["normalized_count"] = adata_test.to_df().values

    # SpaceOracle.imbue_adata_with_space(adata_train, spatial_dim=spatial_dim, in_place=True)
    # pcs = SpaceOracle.perform_PCA(adata_train)
    # SpaceOracle.knn_imputation(adata_train, pcs)

    # SpaceOracle.imbue_adata_with_space(adata_test, spatial_dim=spatial_dim, in_place=True)
    # pcs = SpaceOracle.perform_PCA(adata_test)
    # SpaceOracle.knn_imputation(adata_test, pcs)

    estimator = ViTEstimatorV2(adata_train, target_gene='Cd74', layer='normalized_count')
    assert len(estimator.regulators) == 15

    assert np.intersect1d(estimator.regulators, adata_train.var_names).shape[0] == 15