import sys
sys.path.append('src')
from unittest import TestCase
import pytest 
import numpy as np

from spaceoracle.models.estimators import GeoCNNEstimator
from spaceoracle.tools.utils import set_seed



class GeoCNNTest(TestCase):
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
        self.X = np.random.rand(1000, 13)
        self.y = np.random.rand(1000)
        self.xy = np.random.rand(1000, 2)
        self.labels = np.random.randint(0, 5, (1000,))
        self.fixed_losses_42 = [
            0.10351564362645149, 
            0.10366926155984402, 
            0.10088764317333698, 
            0.10325677692890167, 
            0.1028940100222826
        ]

        self.fixed_betas_42 = [
            0.38564208, 
            0.15239574, 
            -0.04679341, 
            -0.05883441, 
            0.2143733, 
            -0.34551695,
            0.01451041, 
            0.01097491, 
            0.5137137, 
            -0.23085317, 
            0.45055434, 
            -0.07826751,
            -0.02533733,
            -0.13435821
        ]
        
    def test_cnn_estimator(self):
        
        estimator = GeoCNNEstimator()
        set_seed(42)

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
        
        for i in range(5):
            self.assertAlmostEqual(estimator.losses[i], self.fixed_losses_42[i])
         
        self.assertEqual(len(estimator.losses), 5)
        
        betas, y_pred = estimator.get_betas(
            self.X, 
            self.xy, 
            self.labels
        )
        
        self.assertEqual(betas.shape, (1000, 14))
        beta_means = betas.mean(0)
        for i in range(14):
            self.assertAlmostEqual(beta_means[i], self.fixed_betas_42[i])
            