from abc import ABC, abstractmethod
import numpy as np
import sys
import gc
import enlighten
import time
import pandas as pd
import pickle
import torch
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

from .tools.network import DayThreeRegulatoryNetwork
from .models.spatial_map import xyc2spatial
from .models.estimators import ViTEstimatorV2, ViT, device



class Oracle(ABC):
    
    def __init__(self, adata):
        self.adata = adata.copy()
        self.gene2index = dict(zip(self.adata.var_names, range(len(self.adata.var_names))))
        
@dataclass
class BetaOutput:
    betas: np.ndarray
    regulators: List[str]
    target_gene: str
    target_gene_index: int
    regulators_index: List[int]


class SpaceOracle(Oracle):

    def __init__(self, adata, annot='rctd_cluster', init_betas='ols', 
    max_epochs=10, spatial_dim=64, learning_rate=3e-4, batch_size=32, rotate_maps=True, 
    regularize=False, n_patches=2, n_heads=8, n_blocks=4, hidden_d=16):
        
        super().__init__(adata)
        self.annot = annot
        self.spatial_dim = spatial_dim
        self.imbue_adata_with_space(
            self.adata, 
            annot=self.annot,
            spatial_dim=self.spatial_dim,
            in_place=True
        )
        
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        

        self.estimator_models = {}
        self.regulators = {}

        self.genes = list(self.adata.var_names)
        self.trained_genes = []

        # self.losses = pd.DataFrame(columns=genes)

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.genes), 
            desc='Estimating betas', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=max_epochs, 
            desc='Training', 
            unit='epochs',
            color='red',
            autorefresh=True,
        )


        for i in self.genes:
            gene_bar.desc = f'Estimating betas for {i}'
            gene_bar.refresh()

            estimator = ViTEstimatorV2(self.adata, target_gene=i)

            if len(estimator.regulators) > 0:

                estimator.fit(
                    annot=self.annot, 
                    max_epochs=max_epochs, 
                    learning_rate=learning_rate, 
                    spatial_dim=self.spatial_dim,
                    batch_size=batch_size,
                    init_betas=init_betas,
                    mode='train_test',
                    rotate_maps=rotate_maps,
                    regularize=regularize,
                    n_patches=n_patches, 
                    n_heads=n_heads, 
                    n_blocks=n_blocks, 
                    hidden_d=hidden_d,
                    pbar=train_bar
                )

                model, regulators, target_gene = estimator.export()
                assert target_gene == i

                with open(f'./models/{target_gene}_estimator.pkl', 'wb') as f:
                    pickle.dump({'model': model, 'regulators': regulators}, f)
                    self.trained_genes.append(target_gene)

            gene_bar.update()
            train_bar.count = 0
            train_bar.start = time.time()


    def load_estimator(self, gene):
        assert gene in self.trained_genes
        with open(f'./models/{gene}_estimator.pkl', 'rb') as f:
            return pickle.load(f)

    @torch.no_grad()
    def _get_betas(self, adata, target_gene):
        assert target_gene in self.trained_genes
        assert target_gene in adata.var_names
        assert self.annot in adata.obs.columns
        assert 'spatial_maps' in adata.obsm.keys()

        estimator_dict = self.load_estimator(target_gene)
        estimator_dict['model'].eval()

        input_spatial_maps = torch.from_numpy(adata.obsm['spatial_maps']).float().to(device)
        input_cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long().to(device)

        return BetaOutput(
            betas=estimator_dict['model'](input_spatial_maps, input_cluster_labels).cpu().numpy(),
            regulators=estimator_dict['regulators'],
            target_gene=target_gene,
            target_gene_index=self.gene2index[target_gene],
            regulators_index=[self.gene2index[regulator] for regulator in estimator_dict['regulators']]
        )


    def _update_sparse_tensor(self, sparse_tensor, beta_output):
        assert isinstance(beta_output, BetaOutput)

        for k in range(len(beta_output.regulators_index)):
            indices = torch.tensor(
                [[
                    beta_output.regulators_index[k], 
                    beta_output.target_gene_index, i] 
                        for i in range(beta_output.betas.shape[0])
                ], dtype=torch.long)

            values = beta_output.betas[:, k+1]
            new_sparse_tensor = torch.sparse_coo_tensor(indices.t(), values, sparse_tensor.size())
            sparse_tensor = sparse_tensor + new_sparse_tensor

        return sparse_tensor








    def get_coef_matrix(self, adata):
        num_genes = len(adata.var_names)
        indices = torch.empty((3, 0), dtype=torch.long)
        values = torch.empty(0)
        
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (num_genes, num_genes, adata.shape[0]))

        for gene in tqdm(self.trained_genes):
            beta_out = self._get_betas(adata, gene) # cell x beta+1
            sparse_tensor = self._update_sparse_tensor(sparse_tensor, beta_out) # gene x gene x cell

        return sparse_tensor


    def perturb(self, gene_mtx, sparse_tensor, n_propagation=3):
        assert sparse_tensor.shape == (gene_mtx.shape[1], gene_mtx.shape[1], gene_mtx.shape[0])
        
        simulation_input = gene_mtx.copy()

        for i in [74]:
            simulation_input[i] = 0

        delta_input = simulation_input - gene_mtx

        delta_simulated = delta_input.copy()
        
        for i in range(n_propagation):
            delta_simulated = torch.concat([torch.sparse.mm(
                    sparse_tensor[..., i], 
                    torch.from_numpy(delta_simulated)[i].view(delta_simulated.shape[1], 1)
                ) for i in tqdm(range(delta_simulated.shape[0]))], dim=1).numpy().reshape(gene_mtx.shape[0], gene_mtx.shape[1])

            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated

        return gem_simulated


    @staticmethod
    def imbue_adata_with_space(adata, annot='rctd_cluster', spatial_dim=64, in_place=False):
        clusters = np.array(adata.obs[annot])
        xy = np.array(adata.obsm['spatial'])

        sp_maps = xyc2spatial(
            xy[:, 0], 
            xy[:, 1], 
            clusters,
            spatial_dim, spatial_dim, 
            disable_tqdm=False
        ).astype(np.float32)

        if in_place:
            adata.obsm['spatial_maps'] = sp_maps
            return

        return sp_maps
    

    

