import sys
sys.path.append('src')
from unittest import TestCase
import pytest 
import numpy as np

from spaceoracle.models.estimators import GeoCNNEstimator


class GeoCNNTest(TestCase):
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
        self.X = np.random.rand(1000, 13)
        self.y = np.random.rand(1000)
        self.xy = np.random.rand(1000, 2)
        self.labels = np.random.randint(0, 5, (1000,))
        
    def test_cnn_estimator(self):
        estimator = GeoCNNEstimator()

        estimator.fit(
            self.X, 
            self.y, 
            self.xy, 
            labels = self.labels,
            init_betas='ols', 
            max_epochs=5, 
            learning_rate=3e-4, 
            spatial_dim=8,
            in_channels=len(np.unique(self.labels)),
            mode = 'train_test'
        )
        
        print(estimator.beta_init)
        
        betas, y_pred = estimator.get_betas(
            self.X, 
            self.xy, 
            self.labels
        )
        
        self.assertEqual(betas.shape, (1000, 14))
        