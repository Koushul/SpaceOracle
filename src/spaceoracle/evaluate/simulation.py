import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import sys
sys.path.append('/Users/koush/Projects/SpaceOracle/src')

from spaceoracle.models.estimators import GeoCNNEstimator

gt = pd.read_csv('./fixtures/simulated_sinusoidals.csv')

X = gt[['x1', 'x2']].values
y = gt['y'].values
xy = gt[['latitude', 'longitude']].values
c  = np.zeros((len(y), 1))

estimator = GeoCNNEstimator()

estimator.fit(
    X, 
    y, 
    xy, 
    labels = c,
    init_betas='ols', 
    max_epochs=10, 
    learning_rate=3e-3, 
    spatial_dim=64,
    in_channels=1,
    mode = 'train_test',
)

betas, y_pred = estimator.get_betas(X, xy, c)

print(
    pearsonr(betas[:, 0], gt.beta0).statistic, 
    pearsonr(betas[:, 1], gt.beta1).statistic, 
    pearsonr(betas[:, 2], gt.beta2).statistic
)