from abc import ABC, abstractmethod
import numpy as np
import sys
import gc
import enlighten
import time

from .tools.network import DayThreeRegulatoryNetwork
from .models.spatial_map import xyc2spatial
from .models.estimators import ViTEstimatorV2, ViT

class Oracle(ABC):
    
    def __init__(self, adata):
        self.adata = adata.copy()
        
       

class SpaceOracle(Oracle):

    def __init__(self, adata, anot='rctd_cluster', max_epochs=10, spatial_dim=64):
        super().__init__(adata)
        self.anot = anot
        self.spatial_dim = spatial_dim
        self.imbue_adata_with_space(
            self.adata, 
            anot=self.anot,
            spatial_dim=self.spatial_dim,
            in_place=True
        )
        
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        

        self.estimator_models = {}
        self.regulators = {}

        genes = [
            'Uvrag', 'Secisbp2l', 'Nr4a2', 'Dhx40', 'Pcca',
            'Brf1', 'Maea', 'Mllt11', 'Gpank1', 'Ffar2',
            'Igfbp4', 'Nbr1', 'Unc93b1', 'Rin2', 'Mrps26',
            'Malat1', 'Rps20'
        ]

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(genes), 
            desc='Estimating betas', 
            unit='genes'
        )

        train_bar = _manager.counter(
            total=max_epochs, 
            desc='Training', 
            unit='epochs'
        )


        for i in genes:
            self.estimator_models[i] = {}

            estimator = ViTEstimatorV2(self.adata, target_gene=i)

            estimator.fit(
                annot='rctd_cluster', 
                max_epochs=max_epochs, 
                learning_rate=3e-4, 
                spatial_dim=self.spatial_dim,
                batch_size=32,
                init_betas='co',
                mode='train_test',
                rotate_maps=True,
                regularize=False,
                n_patches=16, 
                n_heads=4, 
                n_blocks=3, 
                hidden_d=16,
                pbar=train_bar
            )

            model, regulators, target_gene = estimator.export()
            self.estimator_models[target_gene]['model'] = model
            self.estimator_models[target_gene]['regulators'] = regulators

            gene_bar.desc = f'Estimating betas for {target_gene}'
            gene_bar.update()


            train_bar.count = 0
            train_bar.start = time.time()


    def estimate_betas(self):
        NotImplementedError



    

    @staticmethod
    def imbue_adata_with_space(adata, anot='rctd_cluster', spatial_dim=64, in_place=False):
        clusters = np.array(adata.obs[anot])
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
    

    

