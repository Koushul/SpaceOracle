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
from scipy.stats import norm as normal
import scanpy as sc
import math
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
import warnings
from sklearn.linear_model import Ridge

from spaceoracle.models.probabilistic_estimators import ProbabilisticPixelAttention

from typing import Tuple, Optional, List
from numba import jit
from velocyto.estimation import (colDeltaCor, colDeltaCorLog10,
                                 colDeltaCorLog10partial, colDeltaCorpartial,
                                 colDeltaCorSqrt, colDeltaCorSqrtpartial)

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
    
    def plot_pca(self, dim: List[int]=[0, 1, 2], elev: float=60, azim: float=-140) -> None:
        """Plot 3d PCA
        """

        # update color information
        col_dict = _get_clustercolor_from_anndata(adata=self.adata,
                                                  cluster_name=self.cluster_column_name,
                                                  return_as="dict")
        self.colorandum = np.array([col_dict[i] for i in self.adata.obs[self.cluster_column_name]])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pcs[:, dim[0]],
                   self.pcs[:, dim[1]],
                   self.pcs[:, dim[2]],
                   c=self.colorandum)
        ax.view_init(elev=elev, azim=azim)


    def estimate_transition_prob(self,
                                 n_neighbors: int=None,
                                 knn_random: bool=True, sampled_fraction: float=0.3,
                                 sampling_probs: Tuple[float, float]=(0.5, 0.1),
                                 n_jobs: int=4, threads: int=None, calculate_randomized: bool=True,
                                 random_seed: int=15071990, cell_idx_use=None) -> None:
        """Use correlation to estimate transition probabilities for every cells to its embedding neighborhood

        Arguments
        ---------
        embed: str, default="ts"
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
        transform: str, default="sqrt"
            The transformation that is applies on the high dimensional space.
            If None the raw data will be used

        n_sight: int, default=None (also n_neighbors)
            The number of neighbors to take into account when performing the projection
        knn_random: bool, default=True
            whether to random sample the neighborhoods to speedup calculation
        sampling_probs: Tuple, default=(0.5, 1)
        max_dist_embed: float, default=None
            CURRENTLY NOT USED
            The maximum distance allowed
            If None it will be set to 0.25 * average_distance_two_points_taken_at_random
        n_jobs: int, default=4
            number of jobs to calculate knn
            this only applies to the knn search, for the more time consuming correlation computation see threads
        threads: int, default=None
            The threads will be used for the actual correlation computation by default half of the total.
        calculate_randomized: bool, default=True
            Calculate the transition probabilities with randomized residuals.
            This can be plotted downstream as a negative control and can be used to adjust the visualization scale of the velocity field.
        random_seed: int, default=15071990
            Random seed to make knn_random mode reproducible

        Returns
        -------
        """

        # numba_random_seed(random_seed)

        X = _adata_to_matrix(self.adata, "imputed_count")  # [:, :ndims]
        delta_X = _adata_to_matrix(self.adata, "delta_X")
        embedding = self.adata.obsm[self.embedding_name]
        self.embedding = embedding

        if n_neighbors is None:
            n_neighbors = int(self.adata.shape[0] / 5)


        if knn_random:
            np.random.seed(random_seed)
            self.corr_calc = "knn_random"


            if calculate_randomized:
                delta_X_rndm = np.copy(delta_X)
                permute_rows_nsign(delta_X_rndm)


            if cell_idx_use is None:
                nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
                nn.fit(embedding)  # NOTE should support knn in high dimensions
                self.embedding_knn = nn.kneighbors_graph(mode="connectivity")

            else:
                self.embedding_knn = calculate_embedding_knn_with_cell_idx(embedding_original=self.embedding,
                                                                           cell_idx_use=cell_idx_use,
                                                                           n_neighbors=n_neighbors,
                                                                           n_jobs=n_jobs)

            # Pick random neighbours and prune the rest
            neigh_ixs = self.embedding_knn.indices.reshape((-1, n_neighbors + 1))
            p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
            p = p / p.sum()

            # There was a problem of API consistency because the random.choice can pick the diagonal value (or not)
            # resulting self.corrcoeff with different number of nonzero entry per row.
            # Not updated yet not to break previous analyses
            # Fix is substituting below `neigh_ixs.shape[1]` with `np.arange(1,neigh_ixs.shape[1]-1)`
            # I change it here since I am doing some breaking changes
            sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                                      size=(int(sampled_fraction * (n_neighbors + 1)),),
                                                      replace=False,
                                                      p=p) for i in range(neigh_ixs.shape[0])], 0)
            self.sampling_ixs = sampling_ixs
            neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
            nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
            self.embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                                    neigh_ixs.ravel(),
                                                    np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                                   shape=(neigh_ixs.shape[0],
                                                          neigh_ixs.shape[0]))


            ###
            ###
            self.corrcoef = colDeltaCorpartial(X, delta_X, neigh_ixs, threads=threads)
            if calculate_randomized:
                self.corrcoef_random = colDeltaCorpartial(X, delta_X_rndm, neigh_ixs, threads=threads)
            ######

            if np.any(np.isnan(self.corrcoef)):
                self.corrcoef[np.isnan(self.corrcoef)] = 1
            if calculate_randomized:
                np.fill_diagonal(self.corrcoef_random, 0)
                if np.any(np.isnan(self.corrcoef_random)):
                    self.corrcoef_random[np.isnan(self.corrcoef_random)] = 1
        else:
            self.corr_calc = "full"

            if calculate_randomized:
                delta_X_rndm = np.copy(delta_X)
                permute_rows_nsign(delta_X_rndm)

            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
            nn.fit(embedding)
            self.embedding_knn = nn.kneighbors_graph(mode="connectivity")

            #####
            self.corrcoef = colDeltaCor(X, delta_X, threads=threads)
            if calculate_randomized:
                self.corrcoef_random = colDeltaCor(X, delta_X_rndm, threads=threads)

            #####
            np.fill_diagonal(self.corrcoef, 0)
            if calculate_randomized:
                np.fill_diagonal(self.corrcoef_random, 0)

    def calculate_embedding_shift(self, sigma_corr: float=0.05) -> None:
        """Use the transition probability to project the velocity direction on the embedding

        Arguments
        ---------
        sigma_corr: float, default=0.05
            the kernel scaling

        Returns
        -------
        Nothing but it creates the following attributes:
        transition_prob: np.ndarray
            the transition probability calculated using the exponential kernel on the correlation coefficient
        delta_embedding: np.ndarray
            The resulting vector
        """
        # Kernel evaluation

        # NOTE maybe sparse matrix here are slower than dense
        # NOTE if knn_random this could be made much faster either using sparse matrix or neigh_ixs
        self.transition_prob = np.exp(self.corrcoef / sigma_corr) * self.embedding_knn.A  # naive
        self.transition_prob /= self.transition_prob.sum(1)[:, None]
        if hasattr(self, "corrcoef_random"):
            self.transition_prob_random = np.exp(self.corrcoef_random / sigma_corr) * self.embedding_knn.A  # naive
            self.transition_prob_random /= self.transition_prob_random.sum(1)[:, None]

        unitary_vectors = self.embedding.T[:, None, :].astype(np.float16) - self.embedding.T[:, :, None].astype(np.float16)  # shape (2,ncells,ncells)
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)  # divide by L2
            np.fill_diagonal(unitary_vectors[0, ...], 0)  # fix nans
            np.fill_diagonal(unitary_vectors[1, ...], 0)

        self.delta_embedding = (self.transition_prob * unitary_vectors).sum(2)
        self.delta_embedding -= (self.embedding_knn.A * unitary_vectors).sum(2) / self.embedding_knn.sum(1).A.T
        self.delta_embedding = self.delta_embedding.T


        if hasattr(self, "corrcoef_random"):
            self.delta_embedding_random = (self.transition_prob_random * unitary_vectors).sum(2)
            self.delta_embedding_random -= (self.embedding_knn.A * unitary_vectors).sum(2) / self.embedding_knn.sum(1).A.T
            self.delta_embedding_random = self.delta_embedding_random.T


    def calculate_grid_arrows(self, smooth: float=0.5, steps: Tuple=(40, 40),
                              n_neighbors: int=100, n_jobs: int=4, xylim: Tuple=((None, None), (None, None))) -> None:
        """Calculate the velocity using a points on a regular grid and a gaussian kernel

        Note: the function should work also for n-dimensional grid

        Arguments
        ---------
        embed: str, default=embedding
            The name of the attribute containing the embedding. It will be retrieved as getattr(self, embed)
            The difference vector is getattr(self, 'delta' + '_' + embed)
        smooth: float, smooth=0.5
            Higher value correspond to taking in consideration further points
            the standard deviation of the gaussian kernel is smooth * stepsize
        steps: tuple, default
            the number of steps in the grid for each axis
        n_neighbors:
            number of neighbors to use in the calculation, bigger number should not change too much the results..
            ...as soon as smooth is small
            Higher value correspond to slower execution time
        n_jobs:
            number of processes for parallel computing
        xymin:
            ((xmin, xmax), (ymin, ymax))

        Returns
        -------
        Nothing but it sets the attributes:
        flow_embedding: np.ndarray
            the coordinates of the embedding
        flow_grid: np.ndarray
            the gridpoints
        flow: np.ndarray
            vector field coordinates
        flow_magnitude: np.ndarray
            magnitude of each vector on the grid
        total_p_mass: np.ndarray
            density at each point of the grid

        """
        embedding = self.embedding
        delta_embedding = getattr(self, f"delta_embedding")

        if hasattr(self, "corrcoef_random"):
            delta_embedding_random = getattr(self, f"delta_embedding_random")

        # Prepare the grid
        grs = []
        for dim_i in range(embedding.shape[1]):
            m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])

            if xylim[dim_i][0] is not None:
                m = xylim[dim_i][0]
            if xylim[dim_i][1] is not None:
                M = xylim[dim_i][1]

            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        nn.fit(embedding)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)

        std = np.mean([(g[1] - g[0]) for g in grs])
        # isotropic gaussian kernel
        gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
        self.total_p_mass = gaussian_w.sum(1)

        UZ = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, self.total_p_mass)[:, None]  # weighed average
        magnitude = np.linalg.norm(UZ, axis=1)
        # Assign attributes
        self.flow_embedding = embedding
        self.flow_grid = gridpoints_coordinates
        self.flow = UZ
        self.flow_norm = UZ / np.percentile(magnitude, 99.5)
        self.flow_norm_magnitude = np.linalg.norm(self.flow_norm, axis=1)

        if hasattr(self, "corrcoef_random"):
            UZ_rndm = (delta_embedding_random[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, self.total_p_mass)[:, None]  # weighed average
            magnitude_rndm = np.linalg.norm(UZ, axis=1)
            # Assign attributes
            self.flow_rndm = UZ_rndm
            self.flow_norm_rndm = UZ_rndm / np.percentile(magnitude_rndm, 99.5)
            self.flow_norm_magnitude_rndm = np.linalg.norm(self.flow_norm_rndm, axis=1)

    def calculate_p_mass(self, smooth=0.8, n_grid=40, n_neighbors=200, n_jobs=-1):
        self.calculate_grid_arrows(smooth=0.8, steps=(n_grid, n_grid), n_neighbors=n_neighbors, n_jobs=-1)

    def suggest_mass_thresholds(self, n_suggestion=12, s=1, n_col=4):

        min_ = self.total_p_mass.min()
        max_ = self.total_p_mass.max()
        suggestions = np.linspace(min_, max_/2, n_suggestion)

        n_rows = math.ceil(n_suggestion / n_col)

        fig, ax = plt.subplots(n_rows, n_col, figsize=[5*n_col, 5*n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)

        row = 0
        col = 0
        for i in range(n_suggestion):

            ax_ = ax[row, col]

            col += 1
            if col == n_col:
                col = 0
                row += 1

            idx = self.total_p_mass > suggestions[i]

                #ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax_.scatter(self.embedding[:, 0], self.embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(self.flow_grid[idx, 0],
                       self.flow_grid[idx, 1],
                       c="black", s=s)
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")


    def calculate_mass_filter(self, min_mass=0.01, plot=False):

        self.min_mass = min_mass
        self.mass_filter = (self.total_p_mass < min_mass)

        if plot:
            fig, ax = plt.subplots(figsize=[5,5])

            #ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c="lightgray", s=10)
            ax.scatter(self.flow_grid[~self.mass_filter, 0],
                       self.flow_grid[~self.mass_filter, 1],
                       c="black", s=0.5)
            ax.set_title("Grid points selected")
            ax.axis("off")

    ## Get randomized GRN coef to do randomized perturbation simulation
    def calculate_randomized_coef_table(self, random_seed=123):
        "Calculate randomized GRN coef table."

        if hasattr(self, "coef_matrix_per_cluster"):
            coef_matrix_per_cluster_randomized = {}
            for key, val in self.coef_matrix_per_cluster.items():
                coef_matrix_per_cluster_randomized[key] = _shuffle_celloracle_GRN_coef_table(coef_dataframe=val, random_seed=random_seed)
            self.coef_matrix_per_cluster_randomized = coef_matrix_per_cluster_randomized

        if hasattr(self, "coef_matrix"):
            self.coef_matrix_randomized = _shuffle_celloracle_GRN_coef_table(coef_dataframe=self.coef_matrix, random_seed=random_seed)

        if (hasattr(self, "coef_matrix_per_cluster") == False) and (hasattr(self, "coef_matrix") == False):
            print("GRN calculation for simulation is not finished. Run fit_GRN_for_simulation() first.")


    def prepare_markov(self, sigma_D: np.ndarray, sigma_W: np.ndarray, direction: str="forward", cells_ixs: np.ndarray=None) -> None:
        """Prepare a transition probability for Markov process

        Arguments
        ---------
        sigma_D: float
            the standard deviation used on the locality-limiting component
        sigma_W: float
            the standard deviation used on the noise component
        direction: str, default="backwards"
            whether to diffuse forward of backwards
        cells_ixs: np.ndarray, default=None
            Cells to use, if None all the cells will be considered.

        Returns
        -------
        Nothing but it creates the following attributes:
        tr: np.ndarray
            the transition probability matrix

        """
        if cells_ixs is None:
            cells_ixs = np.arange(self.transition_prob.shape[0])

        # NOTE: This implementation is not speed optimized to improve the speed of the implementation:
        # - the C/Fortran contiguity of the transition matrix should be taken into account
        # - a knn implementation would reduce computation
        # - should avoid transformation to and from dense-sparse formats
        if direction == "forward":
            self.tr = np.array(self.transition_prob[cells_ixs, :][:, cells_ixs])
        elif direction == "backwards":
            self.tr = np.array((self.transition_prob[cells_ixs, :][:, cells_ixs]).T, order="C")
        else:
            raise NotImplementedError(f"{direction} is not an implemented direction")
        dist_matrix = squareform(pdist(self.embedding[cells_ixs, :]))
        K_D = gaussian_kernel(dist_matrix, sigma=sigma_D)
        self.tr = self.tr * K_D
        # Fill diagonal with max or the row and sum=1 normalize
        np.fill_diagonal(self.tr, self.tr.max(1))
        self.tr = self.tr / self.tr.sum(1)[:, None]

        K_W = gaussian_kernel(dist_matrix, sigma=sigma_W)
        K_W = K_W / K_W.sum(1)[:, None]
        self.tr = 0.8 * self.tr + 0.2 * K_W
        self.tr = self.tr / self.tr.sum(1)[:, None]
        self.tr = scipy.sparse.csr_matrix(self.tr)

        if hasattr(self, "corrcoef_random"):
            if direction == "forward":
                self.tr_random = np.array(self.transition_prob_random[cells_ixs, :][:, cells_ixs])
            elif direction == "backwards":
                self.tr_random = np.array((self.transition_prob_random[cells_ixs, :][:, cells_ixs]).T, order="C")
            else:
                raise NotImplementedError(f"{direction} is not an implemented direction")
            #dist_matrix = squareform(pdist(self.embedding[cells_ixs, :]))
            #K_D = gaussian_kernel(dist_matrix, sigma=sigma_D)
            self.tr_random = self.tr_random * K_D
            # Fill diagonal with max or the row and sum=1 normalize
            np.fill_diagonal(self.tr_random, self.tr_random.max(1))
            self.tr_random = self.tr_random / self.tr_random.sum(1)[:, None]

            #K_W = gaussian_kernel(dist_matrix, sigma=sigma_W)
            #K_W = K_W / K_W.sum(1)[:, None]
            self.tr_random = 0.8 * self.tr_random + 0.2 * K_W
            self.tr_random = self.tr_random / self.tr_random.sum(1)[:, None]
            self.tr_random = scipy.sparse.csr_matrix(self.tr_random)

    def run_markov(self, starting_p: np.ndarray=None, n_steps: int=2500, mode: str="time_evolution") -> None:
        """Run a Markov process

        Arguments
        ---------
        starting_p: np.ndarray, default=None
            specifies the starting density
            if None is passed an array of 1/self.tr.shape[0] will be created
        n_steps: np.ndarray, default=2500
            Numbers of steps to be performed
        mode: str, default="time_evolution"
            this argument is passed to the Diffusion.diffuse call

        Returns
        -------
        Nothing but it creates the attribute:
        diffused: np.ndarray
            The probability to be found at any of the states
        """
        self.prepare_markov_simulation()

        if starting_p is None:
            starting_p = np.ones(self.tr.shape[0]) / self.tr.shape[0]
        diffusor = Diffusion()
        self.diffused = diffusor.diffuse(starting_p, self.tr, n_steps=n_steps, mode=mode)[0]


    

        
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




class SpaceOracle(Oracle, Oracle_visualization):

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

        return sparse_tensor
    
    def simulate_shift(self, gex_dict={}):
        genes = list(self.adata.to_df().columns)
        gexidx_dict = {genes.index(goi) : v for goi, v in gex_dict.items()}
        coef_matrix = self.get_coef_matrix(self.adata.copy())
        gene_mtx = self.adata.to_df().values

        gem_simulated = self.perturb(gene_mtx, coef_matrix, gex_dict=gexidx_dict) 
        self.adata.layers['delta_X'] = gem_simulated - self.adata.layers["imputed_count"]


    def perturb(self, gene_mtx, sparse_tensor, gex_dict, n_propagation=3):
        assert sparse_tensor.shape == (gene_mtx.shape[1], gene_mtx.shape[1], gene_mtx.shape[0])
        
        simulation_input = gene_mtx.copy()

        for goi, gex in gex_dict.items():
            simulation_input[goi] = gex

        simulation_input[:, target_index] = 0 # ko target gene
        delta_input = simulation_input - gene_mtx # get delta X
        delta_simulated = delta_input.copy() 

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

        simulation_input[target] = 0 # ko target gene
        delta_input = simulation_input - gene_mtx # get delta X
        delta_simulated = delta_input.copy() 

        if self.coef_matrix is None:
            self.coef_matrix = self._get_co_betas()
        
        for i in range(n_propagation):
            delta_simulated = delta_simulated.dot(self.coef_matrix)
            delta_simulated[delta_input != 0] = delta_input
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated

        return gem_simulated



# def knn_distance_matrix(data, metric=None, k=40, mode='connectivity', n_jobs=4):
#     """Calculate a nearest neighbour distance matrix

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
        pass
>>>>>>> 4585c67 (space oracle transitions)
