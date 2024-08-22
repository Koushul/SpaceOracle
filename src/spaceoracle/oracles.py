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
import os
import shutil
import datetime
import re
import glob
from random import shuffle
from sklearn.decomposition import PCA
import warnings
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy import sparse

from .tools.network import DayThreeRegulatoryNetwork
from .models.spatial_map import xyc2spatial
from .models.estimators import ViTEstimatorV2, ViT, device

import pickle
import io

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
        self.pcs = self.perform_PCA(self.adata)
        self.knn_imputation(self.adata, self.pcs)
        self.gene2index = dict(zip(
                self.adata.var_names, 
                range(len(self.adata.var_names))
            ))

    ## canibalized from CellOracle
    @staticmethod
    def perform_PCA(adata, n_components=None, div_by_std=False):
        X = _adata_to_matrix(adata, "normalized_count")

        pca = PCA(n_components=n_components)
        if div_by_std:
            pcs = pca.fit_transform(X.T / X.std(0))
        else:
            pcs = pca.fit_transform(X.T)

        return pcs

    ## canibalized from CellOracle
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

        if balanced:
            bknn = BalancedKNN(k=k, sight_k=b_sight, maxl=b_maxl,
                               metric=metric, mode="distance", n_jobs=n_jobs)
            bknn.fit(space)
            knn = bknn.kneighbors_graph(mode="distance")
        else:

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
    def remaining_genes(self):
        completed_paths = glob.glob(f'{self.model_dir}/*.pkl')
        locked_paths = glob.glob(f'{self.model_dir}/*.lock')
        completed_genes = list(filter(None, map(self.extract_gene_name, completed_paths)))
        locked_genes = list(filter(None, map(self.extract_gene_name, locked_paths)))
        return list(set(self.regulated_genes).difference(set(completed_genes+locked_genes)))

    def create_lock(self, gene):
        assert not os.path.exists(f'{self.model_dir}/{gene}.lock')
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




class SpaceOracle(Oracle):

    def __init__(self, adata, save_dir='./models', annot='rctd_cluster', init_betas='ols', 
    max_epochs=100, spatial_dim=64, learning_rate=3e-4, batch_size=16, rotate_maps=True, 
    regularize=False, n_patches=2, n_heads=8, n_blocks=4, hidden_d=16):
        
        super().__init__(adata)
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        self.save_dir = save_dir

        self.queue = OracleQueue(save_dir, all_genes=self.adata.var_names)

        self.annot = annot
        self.init_betas = init_betas
        self.max_epochs = max_epochs
        self.spatial_dim = spatial_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rotate_maps = rotate_maps
        self.regularize = regularize
        self.n_patches = n_patches
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d

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

            estimator = ViTEstimatorV2(self.adata, target_gene=gene)

            if len(estimator.regulators) == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'{len(self.queue.orphans)} orphans'
                gene_bar.refresh()

                self.queue.create_lock(gene)

                estimator.fit(
                    annot=self.annot, 
                    max_epochs=self.max_epochs, 
                    learning_rate=self.learning_rate, 
                    spatial_dim=self.spatial_dim,
                    batch_size=self.batch_size,
                    init_betas=self.init_betas,
                    mode='train_test',
                    rotate_maps=self.rotate_maps,
                    regularize=self.regularize,
                    n_patches=self.n_patches, 
                    n_heads=self.n_heads, 
                    n_blocks=self.n_blocks, 
                    hidden_d=self.hidden_d,
                    pbar=train_bar
                )

                model, regulators, target_gene = estimator.export()
                assert target_gene == gene

                with open(f'{self.save_dir}/{target_gene}_estimator.pkl', 'wb') as f:
                    pickle.dump({'model': model, 'regulators': regulators}, f)
                    self.trained_genes.append(target_gene)
                    self.queue.delete_lock(gene)
                    del model

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()

    @staticmethod
    def load_estimator(gene, save_dir):
        with open(f'{save_dir}/{gene}_estimator.pkl', 'rb') as f:
            # return pickle.load(f)
            return CPU_Unpickler(f).load().to(device)

    @torch.no_grad()
    def _get_betas(self, adata, target_gene):
        assert target_gene in self.trained_genes
        assert target_gene in adata.var_names
        assert self.annot in adata.obs.columns
        assert 'spatial_maps' in adata.obsm.keys()

        estimator_dict = self.load_estimator(target_gene, self.save_dir)
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
