import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR


adata_train = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data/snrna_human_tonsil.h5ad')


star = SpaceTravLR(
    adata=adata_train,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-4, 
    spatial_dim=64,
    batch_size=512,
    threshold_lambda=1e-8,
    test_mode=False,
    save_dir='/ix/djishnu/shared/djishnu_kor11/models_v2'
)

star.run()

exit()