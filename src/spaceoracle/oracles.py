import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from abc import ABC, abstractmethod
import numpy as np
import sys
import gc
import enlighten
import time
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import pickle
import torch
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import os
import datetime
import re
import glob
import pickle
import io
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import warnings
from sklearn.linear_model import Ridge
<<<<<<< HEAD

from spaceoracle.models.probabilistic_estimators import ProbabilisticPixelAttention

=======
>>>>>>> e470ac6 (separate co functions, get adata_train co links)
from typing import Tuple, Optional, List

from .oracles_co import CellOracle
from .visualizations.oracle_object_visualization import Oracle_visualization

from .tools.network import DayThreeRegulatoryNetwork
from .models.spatial_map import xyc2spatial, xyc2spatial_fast
from .models.estimators import PixelAttention, device
from .models.pixel_attention import NicheAttentionNetwork

from .tools.utils import (
    CPU_Unpickler,
    knn_distance_matrix,
    _adata_to_matrix,
    connectivity_to_weights,
    convolve_by_sparse_weights
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class Oracle(ABC):
    
    def __init__(self, adata):
        assert 'normalized_count' in adata.layers
        self.adata = adata.copy()
        self.adata.layers['normalized_count'] = self.adata.X.copy()
        self.pcs, self.pca = self.perform_PCA(self.adata)
        self.knn_imputation(self.adata, self.pcs)
        self.gene2index = dict(zip(
                self.adata.var_names, 
                range(len(self.adata.var_names))
            ))

    ## cannibalized from CellOracle
    @staticmethod
    def perform_PCA(adata, n_components=None, div_by_std=False):
        X = _adata_to_matrix(adata, "normalized_count")

        pca = PCA(n_components=n_components)
        if div_by_std:
            pcs = pca.fit_transform(X.T / X.std(0))
        else:
            pcs = pca.fit_transform(X.T)

        return pcs, pca

    ## cannibalized from CellOracle
    @staticmethod
    def knn_imputation(adata, pcs, k=None, metric="euclidean", diag=1,
                       n_pca_dims=None, maximum=False,
                       balanced=False, b_sight=None, b_maxl=None,
                       group_constraint=None, n_jobs=8) -> None:
        
        X = _adata_to_matrix(adata, "normalized_count")

        N = adata.shape[0] # cell number

        if k is None:
            k = int(N * 0.025)
        if b_sight is None and balanced:
            b_sight = int(k * 8)
        if b_maxl is None and balanced:
            b_maxl = int(k * 4)

        space = pcs[:, :n_pca_dims]

        # if balanced:
        #     bknn = BalancedKNN(k=k, sight_k=b_sight, maxl=b_maxl,
        #                        metric=metric, mode="distance", n_jobs=n_jobs)
        #     bknn.fit(space)
        #     knn = bknn.kneighbors_graph(mode="distance")
        # else:

        knn = knn_distance_matrix(space, metric=metric, k=k,
                                        mode="distance", n_jobs=n_jobs)
        connectivity = (knn > 0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            connectivity.setdiag(diag)
        knn_smoothing_w = connectivity_to_weights(connectivity)

        Xx = convolve_by_sparse_weights(X, knn_smoothing_w)
        adata.layers["imputed_count"] = Xx.transpose().copy()

        
@dataclass
class BetaOutput:
    betas: np.ndarray
    regulators: List[str]
    target_gene: str
    target_gene_index: int
    regulators_index: List[int]


class OracleQueue:

    def __init__(self, model_dir, all_genes):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.all_genes = all_genes
        self.orphans = []

    @property
    def regulated_genes(self):
        if not self.orphans:
            return self.all_genes
        return list(set(self.all_genes).difference(set(self.orphans)))
    
    def __getitem__(self, index):
        return self.remaining_genes[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_empty:
            raise StopIteration
        return np.random.choice(self.remaining_genes)

    def __len__(self):
        return len(self.remaining_genes)
        
    @property
    def is_empty(self):
        return self.__len__() == 0

    @property
    def completed_genes(self):
        completed_paths = glob.glob(f'{self.model_dir}/*.pkl')
        return list(filter(None, map(self.extract_gene_name, completed_paths)))

    @property
    def remaining_genes(self):
        completed_paths = glob.glob(f'{self.model_dir}/*.pkl')
        locked_paths = glob.glob(f'{self.model_dir}/*.lock')
        completed_genes = list(filter(None, map(self.extract_gene_name, completed_paths)))
        locked_genes = list(filter(None, map(self.extract_gene_name_from_lock, locked_paths)))
        return list(set(self.regulated_genes).difference(set(completed_genes+locked_genes)))

    def create_lock(self, gene):
        # assert not os.path.exists(f'{self.model_dir}/{gene}.lock')
        now = str(datetime.datetime.now())
        with open(f'{self.model_dir}/{gene}.lock', 'w') as f:
            f.write(now)

    def delete_lock(self, gene):
        assert os.path.exists(f'{self.model_dir}/{gene}.lock')
        os.remove(f'{self.model_dir}/{gene}.lock')

    def add_orphan(self, gene):
        self.orphans.append(gene)

    @staticmethod
    def extract_gene_name(path):
        match = re.search(r'([^/]+)_estimator\.pkl$', path)
        return match.group(1) if match else None
    
    @staticmethod
    def extract_gene_name_from_lock(path):
        match = re.search(r'([^/]+)\.lock$', path)
        return match.group(1) if match else None



class SpaceOracle(Oracle, Oracle_visualization, CellOracle):

    def __init__(self, adata, save_dir='./models', annot='rctd_cluster', 
    max_epochs=15, spatial_dim=64, learning_rate=3e-4, batch_size=256, rotate_maps=True, 
    layer='imputed_count', alpha=0.05):
        
        super().__init__(adata)
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        self.save_dir = save_dir

        self.queue = OracleQueue(save_dir, all_genes=self.adata.var_names)

        self.annot = annot
        self.max_epochs = max_epochs
        self.spatial_dim = spatial_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rotate_maps = rotate_maps
        self.layer = layer
        self.alpha = alpha

        self.beta_dict = None
        self.coef_matrix = None

        if 'spatial_maps' not in self.adata.obsm:
            self.imbue_adata_with_space(
                self.adata, 
                annot=self.annot,
                spatial_dim=self.spatial_dim,
                in_place=True
            )

        self.estimator_models = {}
        self.regulators = {}

        self.genes = list(self.adata.var_names)
        self.trained_genes = []

    
    def run(self):

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc='Estimating betas', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=self.max_epochs, 
            desc='Training', 
            unit='epochs',
            color='red',
            autorefresh=True,
        )


        while not self.queue.is_empty:
            gene = next(self.queue)

            # estimator = ViTEstimatorV2(self.adata, target_gene=gene)

            # estimator = PixelAttention(
            #     self.adata, target_gene=gene, layer=self.layer)

            estimator = ProbabilisticPixelAttention(
                self.adata, target_gene=gene, layer=self.layer)
            
            if len(estimator.regulators) == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'{len(self.queue.orphans)} orphans'
                gene_bar.refresh()

                if os.path.exists(f'{self.queue.model_dir}/{gene}.lock'):
                    continue

                self.queue.create_lock(gene)

                estimator.fit(
                    annot=self.annot, 
                    max_epochs=self.max_epochs, 
                    learning_rate=self.learning_rate, 
                    spatial_dim=self.spatial_dim,
                    batch_size=self.batch_size,
                    mode='train_test',
                    rotate_maps=self.rotate_maps,
                    alpha=self.alpha,
                    pbar=train_bar
                )

                (model, beta_dists, is_real, regulators, target_gene) = estimator.export()
                assert target_gene == gene

                with open(f'{self.save_dir}/{target_gene}_estimator.pkl', 'wb') as f:
                    pickle.dump(
                        {
                            'model': model.state_dict(), 
                            'regulators': regulators,
                            'beta_dists': beta_dists,
                            'is_real': is_real,
                        }, 
                        f
                    )
                    self.trained_genes.append(target_gene)
                    self.queue.delete_lock(gene)
                    del model

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()

    @staticmethod
    def load_estimator(gene, spatial_dim, nclusters, save_dir):
        with open(f'{save_dir}/{gene}_estimator.pkl', 'rb') as f:
            loaded_dict =  CPU_Unpickler(f).load()

            model = NicheAttentionNetwork(
                len(loaded_dict['regulators']), 
                nclusters, 
                spatial_dim
            )
            model.load_state_dict(loaded_dict['model'])

            loaded_dict['model'] = model

        return loaded_dict
    
    
    @torch.no_grad()
    def _get_betas(self, adata, target_gene):
        assert target_gene in adata.var_names
        assert self.annot in adata.obs.columns
        assert 'spatial_maps' in adata.obsm.keys()
        nclusters = len(np.unique(adata.obs[self.annot]))

        estimator_dict = self.load_estimator(target_gene, self.spatial_dim, nclusters, self.save_dir)
        estimator_dict['model'].to(device).eval()
        beta_dists = estimator_dict.get('beta_dists', None)

        input_spatial_maps = torch.from_numpy(adata.obsm['spatial_maps']).float().to(device)
        input_cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long().to(device)
        betas = estimator_dict['model'](input_spatial_maps, input_cluster_labels).cpu().numpy()

        if beta_dists:
            anchors = np.stack([beta_dists[label].mean(0) for label in input_cluster_labels.cpu().numpy()], axis=0)
            betas = betas * anchors

        return BetaOutput(
            betas=betas,
            regulators=estimator_dict['regulators'],
            target_gene=target_gene,
            target_gene_index=self.gene2index[target_gene],
            regulators_index=[self.gene2index[regulator] for regulator in estimator_dict['regulators']]
        )


    def _get_spatial_betas_dict(self):
        beta_dict = {}
        for gene in tqdm(self.queue.completed_genes, desc='Estimating betas globally'):
            beta_dict[gene] = self._get_betas(self.adata, gene)
        
        return beta_dict
    
    def _get_gene_gene_matrix(self, cell_index):
        genes = self.adata.var_names
        gene_gene_matrix = np.zeros((len(genes), len(genes)))

        for i, gene in enumerate(genes):
            _beta_out = self.beta_dict.get(gene, None)
            
            if _beta_out is not None:
                r = np.array(_beta_out.regulators_index)
                gene_gene_matrix[r, i] = _beta_out.betas[cell_index, 1:]

        return gene_gene_matrix


    def _perturb_single_cell(self, gex_delta, cell_index, betas_dict):

        genes = self.adata.var_names
        
        gene_gene_matrix = np.zeros((len(genes), len(genes))) # columns are target genes, rows are regulators

        for i, gene in enumerate(genes):
            _beta_out = betas_dict.get(gene, None)
            
            if _beta_out is not None:
                r = np.array(_beta_out.regulators_index)
                gene_gene_matrix[r, i] = _beta_out.betas[cell_index, 1:]

<<<<<<< HEAD
        return sparse_tensor
    
    def simulate_shift(self, gex_dict={}):
        genes = list(self.adata.to_df().columns)
        gexidx_dict = {genes.index(goi) : v for goi, v in gex_dict.items()}
        coef_matrix = self.get_coef_matrix(self.adata.copy())
        gene_mtx = self.adata.to_df().values
=======
        return gex_delta[cell_index, :].dot(gene_gene_matrix)


    def simulate_shift(self, perturb_condition={}, n_propagation=3):
        '''multi-gene level perturbation'''
>>>>>>> ae1eddd (fix from merge)

        gene_mtx = self.adata.layers['imputed_count']
        simulation_input = gene_mtx.copy()

        for gene, gex in perturb_condition.items():
            target_index = self.gene2index[gene]
            simulation_input[:, target_index] = gex
        
        delta_input = simulation_input - gene_mtx
        delta_simulated = delta_input.copy()

<<<<<<< HEAD
<<<<<<< HEAD
    def perturb(self, gene_mtx, sparse_tensor, gex_dict, n_propagation=3):
        assert sparse_tensor.shape == (gene_mtx.shape[1], gene_mtx.shape[1], gene_mtx.shape[0])
        
=======
        gem_simulated = self.get_gem_simulated(gene_mtx, delta_input, delta_simulated, n_propagation) 
=======
        gem_simulated = self.do_simulation(gene_mtx, delta_input, delta_simulated, n_propagation) 
        self.gem_simulated = gem_simulated
>>>>>>> e470ac6 (separate co functions, get adata_train co links)
        self.adata.layers['delta_X'] = gem_simulated - self.adata.layers["imputed_count"]
    
    def perturb(self, gene_mtx, target, n_propagation=3):
        '''single gene knock-out'''

        assert target in self.adata.var_names

        target_index = self.gene2index[target]  
>>>>>>> ae1eddd (fix from merge)
        simulation_input = gene_mtx.copy()

        for goi, gex in gex_dict.items():
            simulation_input[goi] = gex

        simulation_input[:, target_index] = 0 # ko target gene
        delta_input = simulation_input - gene_mtx # get delta X
        delta_simulated = delta_input.copy() 

        gem_simulated = self.do_simulation(gene_mtx, delta_input, delta_simulated, n_propagation)
        return gem_simulated

    def do_simulation(self, gene_mtx, delta_input, delta_simulated, n_propagation):
        '''perturb helper function'''

        if self.beta_dict is None:
            self.beta_dict = self._get_spatial_betas_dict() # compute betas for all genes for all cells
        
        for n in range(n_propagation):
            _simulated = np.array(
                [self._perturb_single_cell(delta_simulated, i, self.beta_dict) 
                    for i in tqdm(range(self.adata.n_obs), desc=f'Running simulation {n+1}/{n_propagation}')])
            delta_simulated = np.array(_simulated)
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated
        
        assert gem_simulated.shape == gene_mtx.shape

        return gem_simulated


    @staticmethod
    def imbue_adata_with_space(adata, annot='rctd_cluster', spatial_dim=64, in_place=False, method='fast'):
        clusters = np.array(adata.obs[annot])
        xy = np.array(adata.obsm['spatial'])

        if method == 'fast':
            sp_maps = xyc2spatial_fast(
                xyc = np.column_stack([xy, clusters]),
                m=spatial_dim,
                n=spatial_dim,
            ).astype(np.float32)

            # min_vals = np.min(sp_maps, axis=(2, 3), keepdims=True)
            # max_vals = np.max(sp_maps, axis=(2, 3), keepdims=True)
            # denominator = np.maximum(max_vals - min_vals, 1e-15)
            # channel_wise_maps_norm = (sp_maps - min_vals) / denominator
            # sp_maps = channel_wise_maps_norm
                
        else:
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


<<<<<<< HEAD
    def compute_betas(self):
        self.beta_dict = self._get_spatial_betas_dict()
        self.coef_matrix = self._get_co_betas()


    def _get_co_betas(self, alpha=1):

        gem = self.adata.to_df(layer='imputed_count')
        genes = self.adata.var_names
        
        zero_ = pd.Series(np.zeros(len(genes)), index=genes)

        def get_coef(target_gene):
            tmp = zero_.copy()

            reggenes = self.grn.get_regulators(self.adata, target_gene)

            if target_gene in reggenes:
                reggenes.remove(target_gene)
            if len(reggenes) == 0 :
                tmp[target_gene] = 0
                return(tmp)
            
            Data = gem[reggenes]
            Label = gem[target_gene]
            model = Ridge(alpha=alpha, random_state=123)
            model.fit(Data, Label)
            tmp[reggenes] = model.coef_

            return tmp

        li = []
        li_calculated = []
        with tqdm(genes) as pbar:
            for i in pbar:
                if not i in self.queue.completed_genes:
                    tmp = zero_.copy()
                    tmp[i] = 0
                else:
                    tmp = get_coef(i)
                    li_calculated.append(i)
                li.append(tmp)
        coef_matrix = pd.concat(li, axis=1)
        coef_matrix.columns = genes

        return coef_matrix


    def perturb_via_celloracle(self, gene_mtx, target, n_propagation=3):
        
        target_index = self.gene2index[target]  
        simulation_input = gene_mtx.copy()

        simulation_input[target_index] = 0 # ko target gene
        delta_input = simulation_input - gene_mtx # get delta X
        delta_simulated = delta_input.copy() 

        if self.coef_matrix is None:
            self.coef_matrix = self._get_co_betas()
        
        for i in range(n_propagation):
            delta_simulated = delta_simulated.dot(self.coef_matrix)
            # delta_simulated[delta_input != 0] = delta_input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated

        return gem_simulated


<<<<<<< HEAD

# def knn_distance_matrix(data, metric=None, k=40, mode='connectivity', n_jobs=4):
#     """Calculate a nearest neighbour distance matrix

<<<<<<< HEAD
#     Notice that k is meant as the actual number of neighbors NOT INCLUDING itself
#     To achieve that we call kneighbors_graph with X = None
#     """
#     if metric == "correlation":
#         nn = NearestNeighbors(
#             n_neighbors=k, metric="correlation", 
#             algorithm="brute", n_jobs=n_jobs)
#         nn.fit(data)
#         return nn.kneighbors_graph(X=None, mode=mode)
#     else:
#         nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
#         nn.fit(data)
#         return nn.kneighbors_graph(X=None, mode=mode)


# def connectivity_to_weights(mknn, axis=1):
#     if type(mknn) is not sparse.csr_matrix:
#         mknn = mknn.tocsr()
#     return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))

# def convolve_by_sparse_weights(data, w):
#     w_ = w.T
#     assert np.allclose(w_.sum(0), 1)
#     return sparse.csr_matrix.dot(data, w_)


# def _adata_to_matrix(adata, layer_name, transpose=True):
#     if isinstance(adata.layers[layer_name], np.ndarray):
#         matrix = adata.layers[layer_name].copy()
#     else:
#         matrix = adata.layers[layer_name].todense().A.copy()

#     if transpose:
#         matrix = matrix.transpose()

#     return matrix.copy(order="C")
=======
    
=======
>>>>>>> ae1eddd (fix from merge)
def knn_distance_matrix(data, metric=None, k=40, mode='connectivity', n_jobs=4):
    """Calculate a nearest neighbour distance matrix

    Notice that k is meant as the actual number of neighbors NOT INCLUDING itself
    To achieve that we call kneighbors_graph with X = None
    """
    if metric == "correlation":
        nn = NearestNeighbors(n_neighbors=k, metric="correlation", algorithm="brute", n_jobs=n_jobs)
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)
    else:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)


def connectivity_to_weights(mknn, axis=1):
    if type(mknn) is not sparse.csr_matrix:
        mknn = mknn.tocsr()
    return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))

def convolve_by_sparse_weights(data, w):
    w_ = w.T
    assert np.allclose(w_.sum(0), 1), "weight matrix need to sum to one over the columns"
    return sparse.csr_matrix.dot(data, w_)


def _adata_to_matrix(adata, layer_name, transpose=True):
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


## taken directly from CO
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])

## taken directly from CO
def calculate_embedding_knn_with_cell_idx(embedding_original, cell_idx_use, n_neighbors, n_jobs=4):

    """
    Calculate knn graph focusing on a cell population.

    """


    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nn.fit(embedding_original[cell_idx_use, :])  # NOTE should support knn in high dimensions
    embedding_knn = nn.kneighbors_graph(mode="connectivity")

    #print(embedding_knn.indices.max())

    indices_in_original_emb = cell_idx_use[embedding_knn.indices]
    neigh_ixs = np.zeros((embedding_original.shape[0], n_neighbors + 1))
    neigh_ixs[cell_idx_use, :] = indices_in_original_emb.reshape((-1, n_neighbors + 1))

    nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
    embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                      neigh_ixs.ravel(),
                                      np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                      shape=(neigh_ixs.shape[0],
                                             neigh_ixs.shape[0]))
    return embedding_knn

=======
>>>>>>> e470ac6 (separate co functions, get adata_train co links)
## taken directly from CO
def update_adata(adata):
    # Update Anndata
    # Anndata generated with Scanpy 1.4 or less should be updated with this function
    # This function will be depricated in the future.

    try:
        lo = adata.uns['draw_graph']['params']['layout']
        if isinstance(lo, np.ndarray):
            lo = lo[0]
        adata.uns['draw_graph']['params']['layout'] = lo
    except:
<<<<<<< HEAD
        pass
>>>>>>> 4585c67 (space oracle transitions)
=======
        pass
>>>>>>> e470ac6 (separate co functions, get adata_train co links)
