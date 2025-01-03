
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models.parallel_estimators import received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import Betabase

from .plotting.layout import *
from .plotting.transitions import * 
from .plotting.niche import *

from numba import jit


class Prophet(BaseTravLR):
    def __init__(self, adata, models_dir, annot, annot_labels=None, radius=200):
        
        super().__init__(adata, fields_to_keep=[annot, annot_labels])
        
        self.adata = adata.copy()
        self.annot = annot
        self.save_dir = models_dir
        self.annot_labels = annot_labels
        self.radius = radius

        self.queue = OracleQueue(models_dir, all_genes=self.adata.var_names)
        self.ligands = set()
        self.genes = list(self.adata.var_names)
        self.trained_genes = []
        self.betas_cache = {}
        
        self.goi = None

    def compute_betas(self):
        self.beta_dict = self._get_spatial_betas_dict()
    
    @staticmethod
    def load_betadata(gene, save_dir):
        return pd.read_parquet(f'{save_dir}/{gene}_betadata.parquet')
    
    def _get_wbetas_dict(self, betas_dict, gene_mtx):
        
        gex_df = pd.DataFrame(gene_mtx, index=self.adata.obs_names, columns=self.adata.var_names)

        if len(self.ligands) > 0:
            weighted_ligands = received_ligands(
                xy=self.adata.obsm['spatial'], 
                lig_df=gex_df[list(self.ligands)],
                radius=self.radius
            )
        else:
            weighted_ligands = []

        self.weighted_ligands = weighted_ligands

        for gene, betadata in tqdm(betas_dict.data.items(), total=len(betas_dict), desc='Interactions', disable=len(betas_dict) == 1):
            betas_dict.data[gene].wbetas = self._combine_gene_wbetas(gene, weighted_ligands, gex_df, betadata)

        # for gene, betaoutput in tqdm(betas_dict.items(), total=len(betas_dict), desc='Ligand interactions', disable=len(betas_dict) == 1):
        #     betas_df = self._combine_gene_wbetas(gene, weighted_ligands, gex_df, betaoutput)
        #     betas_dict[gene].wbetas = betas_df

        return betas_dict

    def _combine_gene_wbetas(self, gene, rw_ligands, gex_df, betadata):
        betas_df = betadata.splash(rw_ligands, gex_df)
        return betas_df
        

    def _get_spatial_betas_dict(self):
        bdb = Betabase(self.adata, self.save_dir)
        self.ligands = bdb.ligands_set
        return bdb
    
    def _perturb_single_cell(self, gex_delta, cell_index, betas_dict):

        genes = self.adata.var_names
        
        gene_gene_matrix = np.zeros((len(genes), len(genes))) # columns are target genes, rows are regulators

        for i, gene in enumerate(genes):
            _beta_out = betas_dict.data.get(gene, None)
            
            if _beta_out is not None:
                r = np.array(_beta_out.modulator_gene_indices)
                gene_gene_matrix[r, i] = _beta_out.wbetas.values[cell_index]

        return gex_delta[cell_index, :].dot(gene_gene_matrix)
    

    def perturb(self, target, gene_mtx=None, n_propagation=3, gene_expr=0, cells=None):

        self.goi = target
        
        for key in ['transition_probabilities', 'grid_points', 'vector_field']:
            self.adata.uns.pop(key, None)

        assert target in self.adata.var_names
        
        if gene_mtx is None: 
            gene_mtx = self.adata.layers['imputed_count']

        if isinstance(gene_mtx, pd.DataFrame):
            gene_mtx = gene_mtx.values


        target_index = self.gene2index[target]  
        simulation_input = gene_mtx.copy()

        # perturb target gene
        if cells is None:
            simulation_input[:, target_index] = gene_expr   
        else:
            simulation_input[cells, target_index] = gene_expr
        
        delta_input = simulation_input - gene_mtx       # get delta X
        delta_simulated = delta_input.copy() 

        if self.beta_dict is None:
            print('Computing beta_dict')
            self.beta_dict = self._get_spatial_betas_dict() # compute betas for all genes for all cells

        for n in range(n_propagation):

            beta_dict = self._get_wbetas_dict(self.beta_dict, gene_mtx + delta_simulated)

            _simulated = np.array(
                [self._perturb_single_cell(delta_simulated, i, beta_dict) 
                    for i in tqdm(
                        range(self.adata.n_obs), 
                        desc=f'Running simulation {n+1}/{n_propagation}')])
            delta_simulated = np.array(_simulated)
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx

        gem_simulated = gene_mtx + delta_simulated
        
        assert gem_simulated.shape == gene_mtx.shape

        # just as in CellOracle, don't allow simulated to exceed observed values
        imputed_count = gene_mtx
        min_ = imputed_count.min(axis=0)
        max_ = imputed_count.max(axis=0)
        gem_simulated = pd.DataFrame(gem_simulated).clip(lower=min_, upper=max_, axis=1).values

        self.adata.layers['simulated_count'] = gem_simulated
        self.adata.layers['delta_X'] = gem_simulated - imputed_count

        # return gem_simulated
    

    def plot_contour_shift(self, seed=1334, savepath=False):
        assert self.adata.layers.get('delta_X') is not None
        contour_shift(self.adata, annot=self.annot_labels, seed=seed, savepath=savepath)

    def plot_betas_goi(self, save_dir=False):
        betas_goi_all = get_modulator_betas(self, self.goi, save_dir=save_dir)
        self.betas_cache[f'betas_{self.goi}'] = betas_goi_all
    
    def plot_beta_neighborhoods(self, goi=None, use_modulators=False, score_thresh=0.3, savepath=False, seed=1334):
        
        if goi is None:
            goi = self.goi

        if use_modulators:
            # Remove coords and cluster labels
            betas = self.beta_dict.data[goi].iloc[:, :-4].values
        else:
            betas = self.betas_cache.get(f'betas_{goi}')
            if betas is None:
                self.plot_betas_goi()
                betas = self.betas_cache[f'betas_{goi}']
        
        show_beta_neighborhoods(
            self, goi, betas, 
            annot=self.annot_labels, 
            score_thresh=score_thresh,
            seed=seed,
            savepath=savepath
        )

    def show_cluster_gex(self, goi=None, embedding='spatial'):
        if goi is None:
            goi = self.goi
        
        compare_gex(self.adata, annot=self.annot_labels, goi=goi, embedding=embedding)

    def show_transitions(self, layout_embedding=None, nn_embedding=None, vector_scale=1,
    n_neighbors=200, n_jobs=1, savepath=False):
            
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = axs.flatten()

        if layout_embedding is None:
            layout_embedding = self.adata.obsm['spatial']
        
        if nn_embedding is None:
            nn_embedding = self.adata.obsm['X_draw_graph_fr']
        
        estimate_transitions_2D(
            adata=self.adata,
            delta_X=self.adata.layers['delta_X'],
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=self.annot_labels,
            vector_scale=vector_scale,
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs, ax=axs[0]
        )

        delta_X_rndm = self.adata.layers['delta_X'].copy()
        permute_rows_nsign(delta_X_rndm)

        estimate_transitions_2D(
            adata=self.adata,
            delta_X=self.adata.layers['delta_X'],
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=self.annot_labels,
            vector_scale=vector_scale,
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs, ax=axs[1]
        )

        fig.suptitle(f"Transition Estimation from {self.goi} KO", fontsize=16)
        axs[0].set_title("Prediction")
        axs[1].set_title("Randomized")
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)

        plt.show()




# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
