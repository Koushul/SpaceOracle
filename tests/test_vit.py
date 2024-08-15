import pytest
import numpy as np
import torch
import anndata as ad
import sys
import os
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

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


