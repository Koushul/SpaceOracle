import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import spaceoracle
import pickle

from spaceoracle.oracles import SpaceOracle
from spaceoracle.tools.network import MouseKidneyRegulatoryNetwork


co_grn = MouseKidneyRegulatoryNetwork()
adata = sc.read_h5ad('.cache/kidney_chip275_dim35')
spatial_dim = 35 

so = SpaceOracle(
    adata=adata,
    annot='cluster', 
    max_epochs=5, 
    learning_rate=7e-4, 
    spatial_dim=spatial_dim,
    batch_size=256,
    rotate_maps=True,
    alpha=0.4,
    co_grn=co_grn
)

so.run()
exit()