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
from spaceoracle.models.probabilistic_estimators import ProbabilisticPixelAttention
from spaceoracle.models.pixel_attention import NicheAttentionNetwork

import anndata as ad

def generate_realistic_data(noise_level=0.1):
    np.random.seed(42)
    adata = ad.read_h5ad('./data/slideseq/day3_1.h5ad')

    grn = DayThreeRegulatoryNetwork()

    regulators = grn.get_regulators(adata, 'Cd74')[:5]

    adata = adata[:, adata.var_names.isin(regulators+['Cd74'])]

    adata = adata[adata.obs['rctd_cluster'].isin([0, 1])]
    adata = adata[:200, :]

    adata.obs['rctd_cluster'] = adata.obs['rctd_cluster'].cat.remove_unused_categories()

    adata.layers['imputed_count'] = adata.X.toarray().copy()
    adata.layers['normalized_count'] = adata.layers['imputed_count'].copy()

    return adata

@pytest.fixture
def mock_adata_with_true_betas():
    return generate_realistic_data()

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_oracle_initialization(mock_adata_with_true_betas):
    adata = mock_adata_with_true_betas
    oracle = Oracle(adata)
    assert 'imputed_count' in oracle.adata.layers
    assert oracle.pcs is None
    assert oracle.gene2index is not None

    del adata.layers['imputed_count']
    adata = mock_adata_with_true_betas
    oracle = Oracle(adata)
    assert 'imputed_count' in oracle.adata.layers
    assert oracle.pcs is not None
    assert oracle.gene2index is not None



def test_oracle_queue_initialization(temp_dir, mock_adata_with_true_betas):
    adata = mock_adata_with_true_betas
    queue = OracleQueue(temp_dir, adata.var_names.tolist())
    assert queue.model_dir == temp_dir
    assert len(queue.all_genes) == adata.n_vars
    assert len(queue.orphans) == 0

def test_oracle_queue_operations(temp_dir):
    genes = ['gene1', 'gene2', 'gene3']
    queue = OracleQueue(temp_dir, genes)

    # Test remaining_genes
    assert set(queue.remaining_genes) == set(genes)

    # Test create_lock and delete_lock
    queue.create_lock('gene1')
    assert 'gene1.lock' in os.listdir(temp_dir)
    assert set(queue.remaining_genes) == {'gene2', 'gene3'}

    queue.delete_lock('gene1')
    assert 'gene1.lock' not in os.listdir(temp_dir)
    assert set(queue.remaining_genes) == set(genes)

    # Test add_orphan
    queue.add_orphan('gene2')
    assert queue.orphans == ['gene2']
    assert set(queue.remaining_genes) == set(genes)-{'gene2'}

    # Test completed_genes
    with open(os.path.join(temp_dir, 'gene1_estimator.pkl'), 'w') as f:
        f.write('dummy')
    assert queue.completed_genes == ['gene1']
    assert set(queue.remaining_genes) == {'gene3'}

def test_space_oracle_initialization(mock_adata_with_true_betas, temp_dir):
    adata = mock_adata_with_true_betas
    space_oracle = SpaceOracle(adata, save_dir=temp_dir)
    assert space_oracle.adata is not None
    assert space_oracle.grn is not None
    assert space_oracle.queue is not None
    assert 'spatial_maps' in space_oracle.adata.obsm

# @pytest.mark.parametrize("estimator_class", [PixelAttention, ProbabilisticPixelAttention])
@pytest.mark.parametrize("estimator_class", [ProbabilisticPixelAttention])
def test_space_oracle_run(mock_adata_with_true_betas, temp_dir, estimator_class):
    adata = mock_adata_with_true_betas
    with patch('spaceoracle.oracles.PixelAttention', MagicMock(return_value=estimator_class(adata, 'Cd74'))):
        space_oracle = SpaceOracle(adata, save_dir=temp_dir, max_epochs=2, batch_size=3)
        print('Ready to run')
        space_oracle.run()

    assert len(space_oracle.queue.completed_genes) > 0
    assert len(space_oracle.trained_genes) > 0
    assert len(os.listdir(temp_dir)) > 0

def test_space_oracle_load_estimator(mock_adata_with_true_betas, temp_dir):
    adata = mock_adata_with_true_betas
    space_oracle = SpaceOracle(adata, save_dir=temp_dir)
    
    # Create a dummy estimator file
    dummy_estimator = {
        'model': NicheAttentionNetwork(2, 3, 64).state_dict(),
        'regulators': ['gene_1', 'gene_2'],
        'beta_dists': {0: np.random.rand(10, 3)},
        'is_real': pd.DataFrame({'gene_1': [True], 'gene_2': [False]}),
    }
    
    with open(os.path.join(temp_dir, 'gene_0_estimator.pkl'), 'wb') as f:
        import pickle
        pickle.dump(dummy_estimator, f)
    
    loaded_dict = space_oracle.load_estimator('gene_0', 64, 3, temp_dir)
    assert 'model' in loaded_dict
    assert 'regulators' in loaded_dict
    assert loaded_dict['regulators'] == ['gene_1', 'gene_2']

def test_space_oracle_get_betas(mock_adata_with_true_betas, temp_dir):
    adata = mock_adata_with_true_betas
    space_oracle = SpaceOracle(adata, save_dir=temp_dir)
    
    # Mock the load_estimator method
    space_oracle.load_estimator = MagicMock(return_value={
        'model': MagicMock(return_value=torch.rand(100, 3)),
        'regulators': [i for i in adata.var_names.tolist() if i != 'Cd74'],
    })

    adata.obsm['spatial_maps'] = np.random.rand(adata.n_obs, 3, 64, 64)
    
    betas = space_oracle._get_betas(adata, 'Cd74')
    assert betas.betas.shape == (100, 3)
    assert betas.regulators == [i for i in adata.var_names.tolist() if i != 'Cd74']
    assert betas.target_gene == 'Cd74'