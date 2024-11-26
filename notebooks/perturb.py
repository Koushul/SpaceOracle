import scanpy as sc
import pandas as pd 
import numpy as np 
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run SpaceTravLR perturbation for gene of interest.")
parser.add_argument('-goi', required=True, type=str, help="Gene of interest")
args = parser.parse_args()

goi = args.goi

import sys 
sys.path.append('../src')

from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import DayThreeRegulatoryNetwork

co_grn = DayThreeRegulatoryNetwork()
adata = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data/day3_lymph_rep_1.h5ad')

assert goi in adata.var_names, f'{goi} not in adata.var_names'


so = SpaceTravLR(
    adata=adata,
    save_dir='/ix/djishnu/shared/djishnu_kor11/models_v2',
    annot='rctd_cluster', 
    grn=co_grn
)

_ = so.perturb(target=goi, n_propagation=3, gene_expr=0)

# with open('.cache/lymph/bdb.pkl', 'wb') as f:
#     pickle.dump(so.beta_dict, f)

np.save(f'.cache/lymph/{goi}_gem_simulated.npy', 
        so.adata.layers['simulated_count'])

# np.savetxt('.cache/lymph/ligands.txt', np.array(list(so.ligands)), fmt='%s')