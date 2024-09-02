from abc import ABC, abstractmethod
from glob import glob
import anndata
import scanpy as sc
import warnings
import numpy as np
from torch.utils.data import Dataset
from ..models.spatial_map import xyc2spatial
from .network import GeneRegulatoryNetwork
from ..tools.utils import deprecated
import torch

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
            self.spatial_maps = xyc2spatial(    
                self.xy[:, 0], 
                self.xy[:, 1], 
                self.clusters,
                self.spatial_dim, 
                self.spatial_dim,
                disable_tqdm=False
            )
            
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


@deprecated('Please use SpatialDataset.load_slideseq instead.')
def load_example_slideseq(path_dir):
    return [(i, anndata.read_h5ad(i)) for i in glob(path_dir + '/*.h5ad')]