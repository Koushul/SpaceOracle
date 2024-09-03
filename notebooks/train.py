import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import spaceoracle

adata_train = sc.read_h5ad(
    '/ihome/ylee/kor11/space/SpaceOracle/notebooks/cache/adata_train.h5ad')

so = spaceoracle.SpaceOracle(
    adata=adata_train,
    annot='rctd_cluster', 
    max_epochs=15, 
    learning_rate=7e-4, 
    spatial_dim=64,
    batch_size=256,
    init_betas='zeros',
    rotate_maps=True,
    cluster_grn=True,
    regularize=True,
)

so.run()

exit()