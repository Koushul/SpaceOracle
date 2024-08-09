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
        self.fixed_losses_42 = [
            0.10123556000845772, 0.09932259789534978, 
            0.10161668487957545, 0.09764352227960314, 
            0.09753778576850891
        ]
        
        self.fixed_betas_42 = [
            0.4623073, 0.1432567, -0.05220379,
            -0.02402032,  0.2354309, -0.38983065,
            0.02520316,  0.02210942,  0.5923713,  
            -0.23329628,  0.51519436, -0.11414553,
            -0.03308007, -0.15320085
        ]
    
    def _train_model(self):
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
        
        return estimator
    
    
    def test_cnn_estimator(self):
        
        for i in range(3):
            estimator = self._train_model()
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
            print(beta_means)
            for i in range(14):
                self.assertAlmostEqual(beta_means[i], self.fixed_betas_42[i])
                