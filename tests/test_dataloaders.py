import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
import anndata
import torch
from spaceoracle.tools.data import SpaceOracleDataset, SpatialDataset
from spaceoracle.models.estimators import ViTEstimatorV2
from torch.utils.data import DataLoader


class TestSpaceOracleDataset(unittest.TestCase):

    def setUp(self):
        # Create a mock AnnData object
        n_obs = 100
        n_vars = 50
        X = np.random.rand(n_obs, n_vars)
        obs = {'rctd_cluster': np.random.choice([0, 1, 2], size=n_obs)}
        var = {'gene_symbols': [f'gene_{i}' for i in range(n_vars)]}
        obsm = {'spatial': np.random.rand(n_obs, 2)}
        
        self.adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)
        self.adata.layers['normalized_count'] = X
        self.adata.var_names = self.adata.var['gene_symbols']

    def test_spaceoracle_dataset(self):
        target_gene = 'gene_0'
        regulators = ['gene_1', 'gene_2', 'gene_3']
        dataset = SpaceOracleDataset(self.adata, target_gene, regulators)

        self.assertEqual(len(dataset), self.adata.shape[0])

        spatial_info, tf_exp, target_gene_exp, cluster_info = dataset[0]

        self.assertIsInstance(spatial_info, torch.Tensor)
        self.assertIsInstance(tf_exp, torch.Tensor)
        self.assertIsInstance(target_gene_exp, torch.Tensor)
        self.assertIsInstance(cluster_info, torch.Tensor)

        self.assertEqual(
            spatial_info.shape, (dataset.n_clusters, dataset.spatial_dim, dataset.spatial_dim))
        self.assertEqual(tf_exp.shape, (len(regulators),))
        self.assertEqual(target_gene_exp.shape, (1,))
        self.assertEqual(cluster_info.shape, ())

    def test_build_dataloaders_from_adata(self):
        
        # Set up test parameters
        target_gene = 'gene_0'
        regulators = ['gene_1', 'gene_2', 'gene_3']
        batch_size = 16
        spatial_dim = 32
        
        # Test train mode
        train_loader, valid_loader = ViTEstimatorV2._build_dataloaders_from_adata(
            self.adata, target_gene, regulators, batch_size=batch_size, 
            mode='train', spatial_dim=spatial_dim
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(valid_loader, DataLoader)
        assert len(train_loader.dataset) == len(valid_loader.dataset) == len(self.adata)



if __name__ == '__main__':
    unittest.main()
