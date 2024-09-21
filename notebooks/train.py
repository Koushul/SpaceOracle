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

del adata_train.obsm['spatial_maps']

so = spaceoracle.SpaceOracle(
    adata=adata_train,
    annot='rctd_cluster', 
    max_epochs=10, 
    learning_rate=7e-4, 
    spatial_dim=64,
    batch_size=512,
    alpha=0.1,
)

so.run()

exit()