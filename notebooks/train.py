import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import spaceoracle
import pickle


# adata_train = sc.read_h5ad(
#     '/ihome/ylee/kor11/space/SpaceOracle/notebooks/cache/adata_train.h5ad')

# so = spaceoracle.SpaceOracle(
#     adata=adata_train,
#     annot='rctd_cluster', 
#     max_epochs=5, 
#     learning_rate=4e-4, 
#     spatial_dim=64,
#     batch_size=256,
#     init_betas='co',
#     rotate_maps=True,
#     cluster_grn=True,
#     regularize=True,
# )


from utils import get_imputed
from spaceoracle.tools.network import HumanLymphRegulatoryNetwork

adata_train = sc.read_h5ad(
    '../data/spaceranger/human-lymph-node-1-0-0-cleaned.h5ad')

grn = HumanLymphRegulatoryNetwork(base_pth='/ix/djishnu/alw399/SpaceOracle/data/')

del adata_train.obsm['spatial_maps']

so = spaceoracle.SpaceOracle(
    adata=adata_train,
    annot='cluster', 
    max_epochs=5, 
    learning_rate=4e-4, 
    spatial_dim=64,
    batch_size=256,
    init_betas='co',
    rotate_maps=True,
    cluster_grn=True,
    regularize=True,
    co_grn=grn
)


so.run()
so.compute_betas()

with open('beta_dict_visium.pickle', 'wb') as f:
    pickle.dump(so.beta_dict, f)

exit()