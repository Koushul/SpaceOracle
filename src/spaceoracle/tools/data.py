from abc import ABC, abstractmethod
from glob import glob
import anndata
import scanpy as sc
import warnings
import numpy as np
from torch.utils.data import Dataset
from ..models.spatial_map import xyc2spatial_fast
from .network import DayThreeRegulatoryNetwork, GeneRegulatoryNetwork
from ..tools.utils import deprecated, gaussian_kernel_2d
import torch
import pandas as pd

# Suppress ImplicitModificationWarning
warnings.simplefilter(action='ignore', category=anndata.ImplicitModificationWarning)

tt = lambda x: torch.from_numpy(x.copy()).float()

class SpatialDataset(Dataset, ABC):

        
    def __len__(self):
        return self.adata.shape[0]
    
    @staticmethod
    def load_slideseq(path):
        assert '.h5ad' in path
        return anndata.read_h5ad(path)
    
    @staticmethod
    def load_visium(path):
        raise NotImplementedError
    
    @staticmethod
    def load_xenium(path):
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, index):
        pass
        
    

class SpaceOracleDataset(SpatialDataset):
    """
    returns spatial_info, tf_exp, target_ene_exp, cluster_info
    """

    def __init__(self, adata, target_gene, regulators, spatial_dim=16, 
    annot='rctd_cluster', layer='imputed_count', rotate_maps=True):

        self.adata = adata
        
        self.target_gene = target_gene
        self.regulators = regulators
        self.layer = layer
        self.spatial_dim = spatial_dim
        self.rotate_maps = rotate_maps
        
        self.X = adata.to_df(layer=layer)[self.regulators].values
        self.y = adata.to_df(layer=layer)[[self.target_gene]].values
        self.clusters = np.array(self.adata.obs[annot])
        self.n_clusters = len(np.unique(self.clusters))
        self.xy = np.array(self.adata.obsm['spatial'])

        if 'spatial_maps' in self.adata.obsm:
            self.spatial_maps = self.adata.obsm['spatial_maps']
        else:
            self.spatial_maps = xyc2spatial_fast(
                xyc = np.column_stack([self.xy, self.clusters]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            ).astype(np.float32)
            
            self.adata.obsm['spatial_maps'] = self.spatial_maps

    def __getitem__(self, index):
        sp_map = self.spatial_maps[index]
        if self.rotate_maps:
            k = np.random.choice([0, 1, 2, 3])
            sp_map = np.rot90(sp_map, k=k, axes=(1, 2))
        spatial_info = torch.from_numpy(sp_map.copy()).float()
        tf_exp = torch.from_numpy(self.X[index].copy()).float()
        target_gene_exp = torch.from_numpy(self.y[index].copy()).float()
        cluster_info = torch.tensor(self.clusters[index]).long()

        assert spatial_info.shape[0] == self.n_clusters
        assert spatial_info.shape[1] == spatial_info.shape[2] == self.spatial_dim

        return spatial_info, tf_exp, target_gene_exp, cluster_info


class LigRecDataset(SpaceOracleDataset):
    def __init__(
            self, adata, target_gene, regulators, ligands, receptors, radius=200,
            spatial_dim=32, annot='rctd_cluster', layer='imputed_count', rotate_maps=True
        ):
        super().__init__(adata, target_gene, regulators, spatial_dim=spatial_dim, 
                                annot=annot, layer=layer, rotate_maps=rotate_maps)
        self.ligands = ligands
        self.receptors = receptors
        self.radius = radius

        # sq.gr.spatial_neighbors(adata, n_neighs=neighbors)

        xy = np.array(self.adata.obsm['spatial']).copy()
        ligX =  self.adata.to_df(layer=self.layer)[self.ligands].values
        recpX =  self.adata.to_df(layer=self.layer)[self.receptors].values


        # @staticmethod
        # def calculate_ligand_receptor_exp(adata, ligands, receptors, radius):
        #     xy = np.array(adata.obsm['spatial'])
        #     ligX = adata.to_df(layer='imputed_count')[ligands].values
        #     recpX = adata.to_df(layer='imputed_count')[receptors].values
            
        #     lr_exp = np.array([((ligX * gaussian_kernel_2d(
        #         c, xy, radius=radius)[:, np.newaxis])
        #         .mean(axis=0) * recpX[ix]) for ix, c in enumerate(xy)])
            
        #     return pd.DataFrame(
        #         lr_exp,
        #         columns=[i[0]+'$'+i[1] for i in zip(ligands, receptors)],
        #         index=adata.obs.index
        #     )


        received_ligands = self.adata.uns['received_ligands'][self.ligands].values
        receptor_expression = self.adata.to_df(layer=self.layer)[self.receptors].values

        self.lr_exp  = received_ligands * receptor_expression
        
        self.adata.uns['ligand_receptor'] = pd.DataFrame(
            self.lr_exp, 
            columns=[i[0]+'$'+i[1] for i in zip(self.ligands, self.receptors)], 
            index=self.adata.obs.index
        )

        assert self.lr_exp.shape[0] == self.adata.shape[0]
        assert self.lr_exp.shape[1] == len(self.ligands) == len(self.receptors)



    def _process_spatial(self, sp_map):
        if self.rotate_maps:
            k = np.random.choice([0, 1, 2, 3])
            sp_map = np.rot90(sp_map, k=k, axes=(1, 2))
        return sp_map


    def __getitem__(self, index):
        sp_map = self.spatial_maps[index]
        sp_map = self._process_spatial(sp_map)
        
        sp_map = tt(sp_map)
        tf_exp = tt(self.X[index])
        lr_exp = tt(self.lr_exp[index])
        target_gene_exp = tt(self.y[index])
        cluster_info = torch.tensor(self.clusters[index]).long()

        assert sp_map.shape[0] == self.n_clusters
        assert sp_map.shape[1] == sp_map.shape[2] == self.spatial_dim

        x_exp = torch.cat([tf_exp, lr_exp], dim=0)

        return sp_map, x_exp, target_gene_exp, cluster_info
