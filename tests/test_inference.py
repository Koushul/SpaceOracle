import glob
import pytest
import numpy as np
import pandas as pd
import anndata
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import torch

from spaceoracle.tools.network import DayThreeRegulatoryNetwork
from spaceoracle.oracles import Oracle, OracleQueue, SpaceOracle
from spaceoracle.models.estimators import PixelAttention
from spaceoracle.models.probabilistic_estimators import ProbabilisticPixelAttention, ProbabilisticPixelModulators
from spaceoracle.models.pixel_attention import NicheAttentionNetwork
import anndata as ad
from test_oracle import temp_dir
import scanpy as sc


def generate_realistic_data(noise_level=0.1):
    np.random.seed(42)
    adata = ad.read_h5ad('./data/slideseq/day3_1.h5ad')
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=5)
    adata = adata[:, (adata.var.highly_variable | adata.var_names.isin(['Pax5', 'Pou2f2']))]
    adata = adata[adata.obs['rctd_cluster'].isin([0, 1])]
    # adata = adata[:200, :]
    adata.obs['rctd_cluster'] = adata.obs['rctd_cluster'].cat.remove_unused_categories()
    adata.layers['imputed_count'] = adata.X.toarray().copy()
    adata.layers['normalized_count'] = adata.layers['imputed_count'].copy()

    return adata

@pytest.fixture
def mock_adata():
    return generate_realistic_data()

def test_space_oracle_inference(mock_adata, temp_dir):
    so = SpaceOracle(
        adata=mock_adata,
        annot='rctd_cluster', 
        save_dir=temp_dir,
        max_epochs=15, 
        learning_rate=7e-4, 
        spatial_dim=64,
        batch_size=256,
        rotate_maps=True,
        test_mode=True
    )

    so.run()

    assert len(so.queue.orphans) > 0
    assert so.load_betadata(
        so.queue.completed_genes[0], save_dir=so.save_dir).shape == (11976, 8)
    
    target = so.queue.completed_genes[0]

    perturbed_matrix_1 = so.perturb(
        so.adata.to_df(layer='imputed_count').values,
        target=target, n_propagation=1
    )


    co_matrix = so.perturb_via_celloracle(
        so.adata.to_df(layer='imputed_count'),
        target=target, n_propagation=1
    )

    assert len(np.where((so.adata.to_df(layer='imputed_count').values - perturbed_matrix_1).sum(0) !=0)[0]) > 0



