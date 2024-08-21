import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sz = lambda x: f'{asizeof(x) / (1024*1024):.4g} MB'

import spaceoracle
from spaceoracle.models.estimators import ViTEstimatorV2, device
from utils import adata_train, adata_test

so = spaceoracle.SpaceOracle(
    adata_train, 
    init_betas='ones', 
    max_epochs=25, 
    learning_rate=3e-4, 
    spatial_dim=64,
    batch_size=128,
    n_patches=2, n_heads=2, n_blocks=4, hidden_d=16
)