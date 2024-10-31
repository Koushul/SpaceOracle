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
    max_epochs=35, 
    learning_rate=1e-3, 
    spatial_dim=64,
    batch_size=512,
    threshold_lambda=1e-4,
    test_mode=True,
    save_dir='/ix/djishnu/shared/djishnu_kor11/'
)

so.run()

exit()