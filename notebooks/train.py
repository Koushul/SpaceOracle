import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import spaceoracle
import pyro

pyro.clear_param_store()

adata_train = sc.read_h5ad(
    '/ihome/ylee/kor11/space/SpaceOracle/notebooks/cache/adata_train.h5ad')


so = spaceoracle.SpaceOracle(
    adata=adata_train,
    annot='rctd_cluster', 
    max_epochs=7, 
    learning_rate=3e-4, 
    spatial_dim=64,
    batch_size=512,
    alpha=0.9,
)

so.run()

exit()