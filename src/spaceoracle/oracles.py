from abc import ABC
import warnings

from scipy import sparse

from spaceoracle.beta import BetaOutput
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import sys
import gc
import enlighten
import time
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import os
import datetime
import re
import glob
import pickle
import io
import warnings
from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .tools.network import DayThreeRegulatoryNetwork
from .tools.knn_smooth import knn_smoothing
from .beta import Betabase
from .models.spatial_map import xyc2spatial, xyc2spatial_fast
from .models.pixel_attention import NicheAttentionNetwork
from .models.parallel_estimators import SpatialCellularProgramsEstimator, received_ligands

from .tools.utils import (
    CPU_Unpickler,
    clean_up_adata,
    knn_distance_matrix,
    _adata_to_matrix,
    connectivity_to_weights,
    convolve_by_sparse_weights,
    # min_max_df
    prune_neighbors
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class BaseTravLR(ABC):
    
    def __init__(self, adata, fields_to_keep=['rctd_cluster', 'rctd_celltypes']):
        assert 'normalized_count' in adata.layers
        self.adata = adata.copy()
        # self.adata.layers['normalized_count'] = self.adata.X.copy()
        self.gene2index = dict(zip(self.adata.var_names, range(len(self.adata.var_names))))
        self.pcs = None
        
        if 'imputed_count' not in self.adata.layers:
            self.pcs = self.perform_PCA(self.adata)
            self.knn_imputation(self.adata, self.pcs, method='MAGIC')

        clean_up_adata(self.adata, fields_to_keep=fields_to_keep)

    ## cannibalized from CellOracle
    @staticmethod
    def perform_PCA(adata, n_components=None, div_by_std=False):
        X = _adata_to_matrix(adata, "normalized_count")

        pca = PCA(n_components=n_components)
        if div_by_std:
            pcs = pca.fit_transform(X.T / X.std(0))
        else:
            pcs = pca.fit_transform(X.T)
        
        n_comps = np.where(np.diff(np.diff(np.cumsum(pca.explained_variance_ratio_))>0.002))[0][0]

        return pcs[:, :n_comps]

    ## cannibalized from CellOracle
    @staticmethod
    def knn_imputation(adata, pcs, k=None, metric="euclidean", diag=1,
                       n_pca_dims=50, maximum=False,
                       balanced=True, b_sight=None, b_maxl=None,
                       method='MAGIC', n_jobs=8) -> None:
        
        supported_methods = ['CellOracle', 'MAGIC', 'knn-smoothing']
        assert method in supported_methods, f'method is not implemented, choose from {supported_methods}'
        
        X = _adata_to_matrix(adata, "normalized_count")

        N = adata.shape[0] # cell number

        if k is None:
            k = int(N * 0.025)
        if b_sight is None and balanced:
            b_sight = int(k * 8)
        if b_maxl is None and balanced:
            b_maxl = int(k * 4)

        n_pca_dims = min(n_pca_dims, pcs.shape[1])
        space = pcs[:, :n_pca_dims]

        if method == 'CellOracle':
            if balanced:
                nn = NearestNeighbors(n_neighbors=b_sight + 1, metric=metric, n_jobs=n_jobs, leaf_size=30)
                nn.fit(space)

                dist, dsi = nn.kneighbors(space, return_distance=True)
                knn = prune_neighbors(dsi, dist, b_maxl)
            
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
            
        elif method == 'MAGIC':
            import magic
            
            X = X.T
            magic_operator = magic.MAGIC()
            X = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
            X_magic = magic_operator.fit_transform(X, genes='all_genes')

            adata.layers['imputed_count'] = X_magic
        
        elif method == 'knn-smoothing':

            d = 10          # n pcs default 10
            dither = 0.03   # default 0.03 
            k = 32          # number of neighbors 

            matrix = adata.layers['raw_count'].T 
            S = knn_smoothing(matrix, k, d=d, dither=dither, seed=1334)

            adata.layers['imputed_count'] = S.T

            
        
        

        



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
        # completed_paths = glob.glob(f'{self.model_dir}/*.pkl')
        completed_paths = glob.glob(f'{self.model_dir}/*.parquet')
        return list(filter(None, map(self.extract_gene_name, completed_paths)))

    @property
    def num_orphans(self):
        return len(glob.glob(f'{self.model_dir}/*.orphan'))
    
    @property
    def agents(self):
        return len(glob.glob(f'{self.model_dir}/*.lock'))
    
    @property
    def remaining_genes(self):
        # completed_paths = glob.glob(f'{self.model_dir}/*.pkl')
        # completed_paths = glob.glob(f'{self.model_dir}/*.csv')
        completed_paths = glob.glob(f'{self.model_dir}/*.parquet')
        locked_paths = glob.glob(f'{self.model_dir}/*.lock')
        orphan_paths = glob.glob(f'{self.model_dir}/*.orphan')
        completed_genes = list(filter(None, map(self.extract_gene_name, completed_paths)))
        locked_genes = list(filter(None, map(self.extract_gene_name, locked_paths)))
        orphan_genes = list(filter(None, map(self.extract_gene_name, orphan_paths)))
        return list(set(self.regulated_genes).difference(set(completed_genes+locked_genes+orphan_genes)))

    def create_lock(self, gene):
        # assert not os.path.exists(f'{self.model_dir}/{gene}.lock')
        now = str(datetime.datetime.now())
        pid = os.getpid()
        with open(f'{self.model_dir}/{gene}.lock', 'w') as f:
            f.write(f'{now} {pid}')

    def delete_lock(self, gene):
        assert os.path.exists(f'{self.model_dir}/{gene}.lock')
        os.remove(f'{self.model_dir}/{gene}.lock')

    def add_orphan(self, gene):
        now = str(datetime.datetime.now())
        pid = os.getpid()
        with open(f'{self.model_dir}/{gene}.orphan', 'w') as f:
            f.write(f'{now} {pid}')
        self.orphans.append(gene)

    @staticmethod
    def extract_gene_name(path):
        patterns = {
            'betadata': r'([^/]+)_betadata\.parquet$',
            'lock': r'([^/]+)\.lock$',
            'orphan': r'([^/]+)\.orphan$'
        }
        for pattern in patterns.values():
            match = re.search(pattern, path)
            if match:
                return match.group(1)
        return None
    

    def __str__(self):
        return f'OracleQueue with {len(self.remaining_genes)} remaining genes'
    
    def __repr__(self):
        return self.__str__()



class SpaceTravLR(BaseTravLR):

    def __init__(self, adata, save_dir='./models', annot='rctd_cluster', grn=None,
    max_epochs=15, spatial_dim=64, learning_rate=3e-4, batch_size=256, rotate_maps=True, 
    layer='imputed_count', alpha=0.05, test_mode=False, species='mouse', 
    threshold_lambda=3e3, tf_ligand_cutoff=0.01, radius=200):
        
        super().__init__(adata, fields_to_keep=[annot])
        if grn is None:
            self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        else: 
            self.grn = grn

        self.save_dir = save_dir
        self.queue = OracleQueue(save_dir, all_genes=self.adata.var_names)

        self.annot = annot
        self.species = species
        self.max_epochs = max_epochs
        self.spatial_dim = spatial_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rotate_maps = rotate_maps
        self.layer = layer
        self.alpha = alpha
        self.threshold_lambda = threshold_lambda
        self.test_mode = test_mode
        self.tf_ligand_cutoff = tf_ligand_cutoff
        self.beta_dict = None
        self.coef_matrix = None
        self.radius = radius

        self.estimator_models = {}
        self.ligands = set()

        self.genes = list(self.adata.var_names)
        self.trained_genes = []


    def watch(self, sleep=20):
        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        try:

            while not self.queue.is_empty:
                if os.path.exists(self.save_dir+'/process.kill'):
                    print('Found death file. Killing process')
                    break

                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'🕵️️ {self.queue.agents} agents'
                gene_bar.refresh()
                time.sleep(sleep)

        except KeyboardInterrupt:
            pass

    
    def run(self):

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=self.adata.shape[0]*self.max_epochs, 
            desc=f'Ready...', unit='cells',
            color='red',
            auto_refresh=True
        )


        while not self.queue.is_empty and not os.path.exists(self.save_dir+'/process.kill'):
            gene = next(self.queue)

            estimator = SpatialCellularProgramsEstimator(
                adata=self.adata,
                target_gene=gene,
                layer=self.layer,
                cluster_annot=self.annot,
                spatial_dim=self.spatial_dim,
                radius=200,
                tf_ligand_cutoff=self.tf_ligand_cutoff,
                grn=self.grn,
                species=self.species
            )
            
            estimator.test_mode = self.test_mode
            
            if len(estimator.regulators) == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'{self.queue.num_orphans} orphans'
                gene_bar.refresh()

                if os.path.exists(f'{self.queue.model_dir}/{gene}.lock'):
                    continue

                self.queue.create_lock(gene)

                estimator.fit(
                    num_epochs=self.max_epochs, 
                    threshold_lambda=self.threshold_lambda, 
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    pbar=train_bar
                )

                estimator.betadata.to_parquet(f'{self.save_dir}/{gene}_betadata.parquet')

                self.trained_genes.append(gene)
                self.queue.delete_lock(gene)

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()

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

    # This is incorrect, needs to be cluster specific
    # Load links from CO trained model
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


