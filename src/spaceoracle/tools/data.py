from abc import ABC, abstractmethod
from glob import glob
import anndata
import scanpy as sc
import warnings
import numpy as np
from torch.utils.data import Dataset
from ..models.spatial_map import xyc2spatial_fast
from .network import DayThreeRegulatoryNetwork, GeneRegulatoryNetwork
from ..tools.utils import deprecated
import torch
import pandas as pd

# Suppress ImplicitModificationWarning
warnings.simplefilter(action='ignore', category=anndata.ImplicitModificationWarning)


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

        # from sklearn.utils import resample
        # unique_clusters, counts = np.unique(self.clusters, return_counts=True)
        # max_count = max(counts)
        # upsampled_X = self.X.copy()
        # upsampled_y = self.y.copy()
        # upsampled_clusters = self.clusters.copy()
        # upsampled_xy = self.xy.copy()

        # for cluster in unique_clusters:
        #     if counts[cluster] < max_count:
        #         indices = np.where(self.clusters == cluster)[0]
        #         upsampled_indices = resample(indices, n_samples=max_count - counts[cluster], replace=True)
        #         upsampled_X = np.vstack((upsampled_X, self.X[upsampled_indices]))
        #         upsampled_y = np.vstack((upsampled_y, self.y[upsampled_indices]))
        #         upsampled_clusters = np.hstack((upsampled_clusters, self.clusters[upsampled_indices]))
        #         upsampled_xy = np.vstack((upsampled_xy, self.xy[upsampled_indices]))

        # self.X = upsampled_X
        # self.y = upsampled_y
        # self.clusters = upsampled_clusters
        # self.xy = upsampled_xy

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
        target_ene_exp = torch.from_numpy(self.y[index].copy()).float()
        cluster_info = torch.tensor(self.clusters[index]).long()

        assert spatial_info.shape[0] == self.n_clusters
        assert spatial_info.shape[1] == spatial_info.shape[2] == self.spatial_dim

        return spatial_info, tf_exp, target_ene_exp, cluster_info


class LigRecDataset(SpaceOracleDataset):
    def __init__(
            self, adata, target_gene, regulators, ligands, receptors, radius=20,
            spatial_dim=16, annot='rctd_cluster', layer='imputed_count', rotate_maps=True
        ):
        super().__init__(adata, target_gene, regulators, spatial_dim=spatial_dim, 
                                annot=annot, layer=layer, rotate_maps=rotate_maps)
        self.ligands = ligands
        self.receptors = receptors
        self.radius = radius

        self.ligX =  adata.to_df(layer=layer)[self.ligands].values
        self.recX =  adata.to_df(layer=layer)[self.receptors].values
        self.dist_matrix = self.compute_distances(adata.obsm['X_spatial'])
        # max number of neighbors within radius
        self.context = np.max([len(np.argwhere(arr < self.radius)) for arr in self.dist_matrix]) 

    def compute_distances(self, coords):
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        return distances

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

        arr = self.dist_matrix[index]
        neighbors = np.argwhere((arr < self.radius) & (arr != 0))
        nneighbors = len(neighbors)
        
        distances = self.xy[neighbors].squeeze() 
        pad_size = self.context - nneighbors
        distX = np.vstack([distances, np.zeros((pad_size, 2))])

        nligands = len(self.ligands)
        nreceptors = len(self.receptors)
        ligX = self.ligX[neighbors]
        ligX = ligX.reshape(nneighbors, nligands)
        ligX = np.vstack([ligX, np.zeros((pad_size, ligX.shape[1]))])

        recX = self.recX[index]

        tf_load = (spatial_info, tf_exp, target_gene_exp, cluster_info)
        lr_load = (torch.from_numpy(distX), torch.from_numpy(ligX).float(), torch.from_numpy(recX).float())
        return tf_load, lr_load