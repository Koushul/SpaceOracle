import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sz = lambda x: f'{asizeof(x) / (1024*1024):.4g} MB'

import spaceoracle
from spaceoracle.models.estimators import ViTEstimatorV2, device
from utils import adata_train, adata_test

so = spaceoracle.SpaceOracle(
    annot='rctd_cluster', 
    max_epochs=13, 
    learning_rate=7e-4, 
    spatial_dim=64,
    batch_size=256,
    init_betas='ones',
    mode='train_test',
    rotate_maps=True,
    cluster_grn=True,
    regularize=False,
    n_patches=4, n_heads=2, n_blocks=4, hidden_d=16
)

so.run()