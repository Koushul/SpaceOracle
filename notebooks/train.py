import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import spaceoracle
from spaceoracle.models.estimators import ViTEstimatorV2, device
# from utils import adata_train, adata_test

adata_train = sc.read_h5ad(
    '/ihome/ylee/kor11/space/SpaceOracle/notebooks/cache/adata_train.h5ad')

so = spaceoracle.SpaceOracle(
    adata=adata_train,
    annot='rctd_cluster', 
    max_epochs=10, 
    learning_rate=7e-4, 
    spatial_dim=64,
    batch_size=256,
    init_betas='ols',
    rotate_maps=True,
    cluster_grn=True,
    regularize=False,
    n_patches=8, n_heads=2, n_blocks=4, hidden_d=16
)

so.run()