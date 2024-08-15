import pytest
import numpy as np
import torch
import anndata as ad
import sys
import os
import pandas as pd

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

# def test_vit_estimator_reproducibility(mock_adata):
#     target_gene = 'gene_0'
#     mock_adata = ad.read_h5ad('/Users/koush/Projects/SpaceOracle/data/slideseq/day3_1.h5ad')
#     mock_adata = mock_adata[mock_adata.obs.rctd_cluster.isin([0, 1, 2]), :]
#     mock_adata = mock_adata[:10, :101]
#     estimator1 = ViTEstimatorV2(mock_adata, mock_adata.var_names[0])
#     estimator2 = ViTEstimatorV2(mock_adata, mock_adata.var_names[0])
    
#     estimator1.fit(annot='rctd_cluster', max_epochs=5)
#     estimator2.fit(annot='rctd_cluster', max_epochs=5)
    
#     betas1, _ = estimator1.get_betas(
#         xy=mock_adata.obsm['spatial'], labels=np.array(mock_adata.obs['rctd_cluster']), spatial_dim=32)
#     betas2, _ = estimator2.get_betas(
#         xy=mock_adata.obsm['spatial'], labels=np.array(mock_adata.obs['rctd_cluster']), spatial_dim=32)
    
#     np.testing.assert_allclose(betas1, betas2, rtol=1e-5, atol=1e-5)


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
