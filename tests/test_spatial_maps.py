import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.datasets import make_regression
from unittest import TestCase
import numpy as np

from spaceoracle.models.spatial_map import xyc2spatial, xyc2spatial_fast

class SpatialMapsTest(TestCase):

    # def test_xyc2spatial(self):
    #     n_samples = np.random.randint(100, 1000)
    #     n_clusters = np.random.randint(2, 10)
    #     m = n = np.random.randint(3, 12)
        
    #     X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1)
    #     labels = np.random.randint(0, n_clusters, (n_samples,))

    #     self.assertEqual(
    #         xyc2spatial(X[:, 0], X[:, 1], labels, m, n).shape, 
    #         (n_samples, n_clusters, m, n)
    #     )

    def test_xyc2spatial_fast(self):
        n_samples = np.random.randint(100, 1000)
        n_clusters = np.random.randint(2, 10)
        m = n = np.random.randint(3, 12)
        
        X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1)
        labels = np.random.randint(0, n_clusters, (n_samples,))

        spatial_maps = xyc2spatial_fast(
            xyc = np.column_stack([X, labels]),
            m=m,
            n=n,
        ).astype(np.float32)

        self.assertEqual(
            spatial_maps.shape, 
            (n_samples, n_clusters, m, n)
        )
        