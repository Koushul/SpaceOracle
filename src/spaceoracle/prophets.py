
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from .models.parallel_estimators import received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import Betabase

from .plotting.layout import *
from .plotting.transitions import * 
from .plotting.niche import *
from .plotting.beta_maps import *

from numba import jit


class Prophet(BaseTravLR):
    def __init__(self, adata, models_dir, annot, annot_labels=None, radius=200):

        if annot_labels == None:
            annot_labels = annot
        
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
        self.gsea_scores = {}
        self.sim_adata = None

        with open('../../data/GSEA_human/h.all.v2024.1.Hs.json', 'r') as f:
            self.gsea_modules = json.load(f)
        # with open('../../data/GSEA_human/c7.immunesigdb.v2024.1.Hs.json', 'r') as f:
        #     gsea_immune = json.load(f)

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
        contour_shift(self.adata, gene=self.goi, annot=self.annot_labels, seed=seed, savepath=savepath)

    def plot_betas_goi(self, goi=None, save_dir=False, use_simulated=False, clusters=[], blur=False):
        '''use_simulated: if True, compute rw_ligands from simulated_count, else from imputed_count'''
        if goi is None:
            goi = self.goi
        betas_goi_all = get_modulator_betas(self, goi, save_dir=save_dir, use_simulated=use_simulated, clusters=clusters, blur=blur)
        self.betas_cache[f'betas_{goi}'] = betas_goi_all
    
    def plot_beta_neighborhoods(self, goi=None, use_modulators=False, score_thresh=0.3, savepath=False, seed=1334):
        
        if goi is None:
            goi = self.goi

        if use_modulators:
            # Remove coords and cluster labels
            assert goi in self.beta_dict.data.keys(), f'{goi} does not have modulators'
            betas = self.beta_dict.data[goi].iloc[:, :-4].values
        else:
            betas = self.betas_cache.get(f'betas_{goi}')
            if betas is None:
                self.plot_betas_goi()
                betas = self.betas_cache[f'betas_{goi}']
        
        labels = show_beta_neighborhoods(
            self, goi, betas, 
            annot=self.annot_labels, 
            score_thresh=score_thresh,
            seed=seed,
            savepath=savepath
        )

        self.adata.obs['beta_neighborhood'] = labels
        self.adata.obs['beta_neighborhood'] = self.adata.obs['beta_neighborhood'].astype('category')

    def plot_beta_map(self, regulator, target_gene, clusters=None, save_dir=False):

        if clusters is None:
            clusters = self.adata.obs[self.annot].value_counts().head(3).index

        beta_data = self.beta_dict.data[target_gene]
        beta_data[self.annot_labels] = self.adata.obs[self.annot_labels]

        ax = plot_spatial(
            df=beta_data,
            plot_for=f'beta_{regulator}',
            target_gene=target_gene,
            clusters=clusters,
            annot=self.annot,
            annot_labels=self.annot_labels,
            with_expr=False
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            savepath = os.path.join(save_dir, f'{target_gene}_beta_{regulator}_map.png')
            plt.savefig(savepath)


    def plot_beta_umap(self, use_modulators=False, seed=1334, n_neighbors=50):

        assert 'beta_neighborhood' in self.adata.obs.columns, f'Run plot_beta_neighborhood() first'

        reducer = umap.UMAP(random_state=seed, n_neighbors=n_neighbors, min_dist=1.0, spread=5.0)
        
        if use_modulators is True:
            X = self.beta_dict.data[self.goi].iloc[:, :-4].values
        else:
            X = self.betas_cache[f'betas_{self.goi}']
        
        umap_coords = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            x=umap_coords[:,0], 
            y=umap_coords[:,1],
            hue=self.adata.obs['beta_neighborhood'].values,
            alpha=0.5,
            s=20,
            ax=ax,
        )
        plt.title(f'Beta UMAP for {self.goi}')

        self.adata.obsm['beta_umap'] = umap_coords

    def show_cluster_gex(self, goi=None, embedding='spatial', annot=None):
        if goi is None:
            goi = self.goi
        
        if annot is None:
            annot = self.annot_labels
        
        compare_gex(self.adata, annot=annot, goi=goi, embedding=embedding)

    def show_transitions(self, layout_embedding=None, nn_embedding=None, vector_scale=1,
    grid_scale=1, annot=None, n_neighbors=200, n_jobs=1, savepath=False):
            
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = axs.flatten()

        if layout_embedding is None:
            layout_embedding = self.adata.obsm['spatial']
        
        if nn_embedding is None:
            nn_embedding = self.adata.obsm['X_draw_graph_fr']
        
        if annot is None:
            annot = self.annot_labels
        
        estimate_transitions_2D(
            adata=self.adata,
            delta_X=self.adata.layers['delta_X'],
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=annot,
            grid_scale=grid_scale,
            vector_scale=vector_scale,
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs, ax=axs[0]
        )

        delta_X_rndm = self.adata.layers['delta_X'].copy()
        permute_rows_nsign(delta_X_rndm)

        estimate_transitions_2D(
            adata=self.adata,
            delta_X=delta_X_rndm,
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=annot,
            grid_scale=grid_scale,
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
    
    
    def create_sim_adata(self):

        assert 'simulated_count' in self.adata.layers.keys(), 'Run perturb first'
        
        self.sim_adata = sc.AnnData(
                X=self.adata.layers['simulated_count'],
                obs=self.adata.obs,
                var=self.adata.var,
                obsm=self.adata.obsm
            )
        return self.sim_adata

    def compute_gsea_scores(self, use_simulated=False, show_spatial=True, savepath=False):

        if use_simulated:
            adata = self.create_sim_adata()
            label= f'simulated_{self.goi}'
        else:
            adata = self.adata.copy()
            adata.X = adata.layers['imputed_count']
            label= 'observed'

        gsea_scores = self.gsea_scores.get(label, None)

        if gsea_scores is None:
        
            gsea_scores = {}

            for mod_name, mod_dict in self.gsea_modules.items():
                gene_list = mod_dict['geneSymbols']
                gene_list = [g for g in gene_list if g in adata.var_names]
                score_name = f'{mod_name}'

                sc.tl.score_genes(adata, gene_list, score_name=score_name, use_raw=False)

                gsea_scores[mod_name] = adata.obs[score_name]
            
            gsea_scores = pd.DataFrame(gsea_scores, columns=self.gsea_modules.keys()).T
            gsea_scores['score_var'] = gsea_scores.var(axis=1)
            self.gsea_scores[label] = gsea_scores.sort_values('score_var', ascending=False)

        if 'observed' in self.gsea_scores.keys():
            modules = list(self.gsea_scores['observed'].head(4).index)
        else:
            modules = list(gsea_scores.head(4).index)

        plot_params = {
            "color": [self.annot_labels] + modules,
            "ncols": 5,
            "show": not savepath,
        }

        if show_spatial:
            plot_params["spot_size"] = 50
            sc.pl.spatial(adata, **plot_params)
        else:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            sc.pl.umap(adata, **plot_params)

        if savepath:
            plt.savefig(savepath)
    





# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
