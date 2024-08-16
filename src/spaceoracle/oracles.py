from abc import ABC, abstractmethod
import numpy as np
import sys
import gc
from pympler import asizeof

from .models.spatial_map import xyc2spatial
from .models.estimators import ViTEstimatorV2

class Oracle(ABC):
    
    def __init__(self, adata):
        self.adata = adata.copy()
        
       

class SpaceOracle(Oracle):

    def __init__(self, adata, anot='rctd_cluster', spatial_dim=64):
        super().__init__(adata)
        self.anot = anot
        self.spatial_dim = spatial_dim
        self.imbue_adata_with_space(
            self.adata, 
            anot=self.anot,
            spatial_dim=self.spatial_dim,
            in_place=True
        )

        self.estimator_models = {}
        self.regulators = {}

        for i in ['Cd74']:
            self.estimator_models[i] = {}

            estimator = ViTEstimatorV2(self.adata, target_gene=i)

            print(asizeof.asizeof(estimator)/(1024*1024))

            estimator.fit(
                annot='rctd_cluster', 
                max_epochs=2, 
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
                hidden_d=16
            )

            model, regulators, target_gene = estimator.export()
            self.estimator_models[target_gene]['model'] = model
            self.estimator_models[target_gene]['regulators'] = regulators



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
    

    

