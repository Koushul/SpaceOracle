import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR


##########################
### LYMPH NODE DATASET ###
##########################

# adata_train = sc.read_h5ad(
#     '/ix/djishnu/shared/djishnu_kor11/training_data/day3_lymph_rep_1.h5ad')

# star = SpaceTravLR(
#     adata=adata_train,
#     annot='rctd_cluster', 
#     max_epochs=200, 
#     learning_rate=5e-4, 
#     spatial_dim=64,
#     batch_size=512,
#     threshold_lambda=1e-8,
#     test_mode=False,
#     save_dir='/ix/djishnu/shared/djishnu_kor11/models_v2'
# )

########################
### SPLEEN SLIDE-SEQ ###
########################

from spaceoracle.tools.network import MouseSpleenRegulatoryNetwork
co_grn = MouseSpleenRegulatoryNetwork()

adata_train = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data/slideseq_spleen.h5ad')

star = SpaceTravLR(
    adata=adata_train,
    annot='clusters', 
    grn=co_grn,
    max_epochs=200, 
    learning_rate=5e-4, 
    spatial_dim=64,
    batch_size=512,
    threshold_lambda=1e-12,
    test_mode=False,
    save_dir='/ix/djishnu/shared/djishnu_kor11/models_spleen'
)


star.run()
exit()