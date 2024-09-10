# -*- coding: utf-8 -*-
import numpy as np                    
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import norm as normal       
from scipy import sparse               
from sklearn.neighbors import NearestNeighbors 
from numba import jit
from velocyto.estimation import (colDeltaCor, colDeltaCorLog10,
                                 colDeltaCorLog10partial, colDeltaCorpartial,
                                 colDeltaCorSqrt, colDeltaCorSqrtpartial)
from typing import Tuple, List
from tqdm import tqdm

import math                            
import matplotlib.pyplot as plt   

from .oracle_utility import _adata_to_df, _get_clustercolor_from_anndata, _linklist2dict
from .tools.utils import _adata_to_matrix


class CellOracle:

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

    def _perturb_co_single_cell(self, gex_delta, cell_index, betas_dict):

        genes = self.adata.var_names

        gene_gene_matrix = [] # columns are target genes, rows are regulators
        cluster = list(self.adata.obs['louvain_annot'])[cell_index]
        beta_dict = betas_dict[cluster]

        for i, gene in enumerate(genes):
            _beta_out = beta_dict.iloc[i]
            gene_gene_matrix.append(_beta_out)

        gene_gene_matrix = np.array(gene_gene_matrix)
        return gex_delta[cell_index, :].dot(gene_gene_matrix)

    def simulate_co_shift(self, perturb_condition={}, n_propagation=3):
        '''adapted function to test against CO results'''
        gene_mtx = self.adata.layers['imputed_count']
        simulation_input = gene_mtx.copy()

        for gene, gex in perturb_condition.items():
            target_index = self.gene2index[gene]
            simulation_input[:, target_index] = gex
        
        delta_input = simulation_input - gene_mtx
        delta_simulated = delta_input.copy()

        if not hasattr(self, 'co_beta_dict'):
            self.co_beta_dict = self.grn.coef_matrix_per_cluster

        # do_simulation
        for n in range(n_propagation):
            _simulated = np.array(
                [self._perturb_co_single_cell(delta_simulated, i, self.co_beta_dict) 
                    for i in tqdm(range(self.adata.n_obs), desc=f'Running simulation {n+1}/{n_propagation}')])
            delta_simulated = np.array(_simulated)
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated
        self.adata.layers['perturbed_co'] = gem_simulated        
        self.adata.layers['delta_X'] = gem_simulated - self.adata.layers["imputed_count"]

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

        unitary_vectors = self.embedding.T[:, None, :] - self.embedding.T[:, :, None]  # shape (2,ncells,ncells)
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
        self.tr = scipy.csr_matrix(self.tr)

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
