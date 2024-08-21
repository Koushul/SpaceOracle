print('init...')

import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sz = lambda x: f'{asizeof(x) / (1024*1024):.4g} MB'

import spaceoracle
from spaceoracle.models.estimators import ViTEstimatorV2, device
from utils import adata_train, adata_test

print(device)

so = spaceoracle.SpaceOracle(
    adata_train, 
    init_betas='ones', 
    spatial_dim=64,
    max_epochs=100, 
    learning_rate=3e-3
)