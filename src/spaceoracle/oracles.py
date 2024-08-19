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

from .tools.network import DayThreeRegulatoryNetwork
from .models.spatial_map import xyc2spatial
from .models.estimators import ViTEstimatorV2, ViT, device



class Oracle(ABC):
    
    def __init__(self, adata):
        self.adata = adata.copy()
        self.gene2index = dict(zip(self.adata.var_names, range(len(self.adata.var_names))))
        
@dataclass
class OracleOutput:
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

        return OracleOutput(
            betas=estimator_dict['model'].forward(input_spatial_maps, input_cluster_labels).cpu().numpy(),
            regulators=estimator_dict['regulators'],
            target_gene=target_gene,
            target_gene_index=self.gene2index[target_gene],
            regulators_index=[self.gene2index[regulator] for regulator in estimator_dict['regulators']]
        )



    def get_coef_matrix(self, adata):
        gem = adata.to_df()
        genes = gem.columns
        all_genes_in_dict = intersect(gem.columns, list(TFdict.keys()))
        zero_ = pd.Series(np.zeros(len(genes)), index=genes)





        NotImplementedError





    

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
    

    

