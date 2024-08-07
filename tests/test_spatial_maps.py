import sys

from sklearn.datasets import make_regression
sys.path.append('../src')

from unittest import TestCase
import numpy as np
import pytest 

from spaceoracle.models.spatial_map import xyc2spatial

class SpatialMapsTest(TestCase):
        
    def test_xyc2spatial(self):
        n_samples = np.random.randint(100, 1000)
        n_clusters = np.random.randint(2, 10)
        m = n = np.random.randint(3, 12)
        
        X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1)
        labels = np.random.randint(0, n_clusters, (n_samples,))

        self.assertEqual(
            xyc2spatial(X[:, 0], X[:, 1], labels, m, n).shape, 
            (n_samples, n_clusters, m, n)
        )