# import celloracle as co
import numpy as np
import pandas as pd
import pickle
import os
import json 
import torch
import networkx as nx 

def expand_paired_interactions(df):
    expanded_rows = []
    for _, row in df.iterrows():
        ligands = row['ligand'].split('_')
        receptors = row['receptor'].split('_')
        
        for ligand in ligands:
            for receptor in receptors:
                new_row = row.copy()
                new_row['ligand'] = ligand
                new_row['receptor'] = receptor
                expanded_rows.append(new_row)
    
    df = pd.DataFrame(expanded_rows)
    
    return df

class GeneRegulatoryNetwork:
    def __init__(self, organism='mouse'):
        if organism == 'mouse':
            # self.data = co.data.load_mouse_scATAC_atlas_base_GRN()
            import os
            data_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data', 'mm9_mouse_atac_atlas_data_TSS.parquet')
            self.data = pd.read_parquet(data_path)
            
    def get_regulators(self, adata, target_gene):
        base_GRN = self.data
        
        df = base_GRN[base_GRN.gene_short_name==target_gene][
            np.intersect1d(adata.var_names, base_GRN[base_GRN.gene_short_name==target_gene].columns)].sum()
        df = df[df!=0]
        
        return df.index.tolist()
        

class CellOracleLinks:
    
    def __init__(self):
        pass

    def get_regulators(self, adata, target_gene, alpha=0.05):
        regulators_with_pvalues = self.get_regulators_with_pvalues(adata, target_gene, alpha)
        grouped_regulators = regulators_with_pvalues.groupby('source').mean()
        filtered_regulators = grouped_regulators[grouped_regulators.index.isin(adata.var_names)]

        return filtered_regulators.index.tolist()
    
    def get_targets(self, adata, tf, alpha=0.05):
        targets_with_pvalues = self.get_targets_with_pvalues(adata, tf, alpha)
        grouped_targets = targets_with_pvalues.groupby('target').mean()
        filtered_targets = grouped_targets[grouped_targets.index.isin(adata.var_names)]

        return filtered_targets.index.tolist()

    def get_regulators_with_pvalues(self, adata, target_gene, alpha=0.05):
        assert target_gene in adata.var_names, f'{target_gene} not in adata.var_names'
        co_links = pd.concat(
            [link_data.query(f'target == "{target_gene}" and p < {alpha}')[['source', 'coef_mean']] 
                for link_data in self.links.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'source.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    def get_targets_with_pvalues(self, adata, tf, alpha=0.05):
        assert tf in adata.var_names, f'{tf} not in adata.var_names'
        co_links = pd.concat(
            [link_data.query(f'source == "{tf}" and p < {alpha}')[['target', 'coef_mean']] 
                for link_data in self.links.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'target.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    @staticmethod
    def get_training_genes(co_links, gene_kos, n_propagation=3):
        grn = nx.DiGraph()
        edges = []

        for cluster, df in co_links.items():
            cluster_edges = [(u, v) for u, v in zip(df['source'], df['target'])]
            edges.extend(cluster_edges)
        
        grn.add_edges_from(edges)
        train_genes = []
        for ko in gene_kos:
            genes = [node for node, distance in nx.single_source_shortest_path_length(
                                                grn, ko, cutoff=n_propagation).items()]
            train_genes.extend(genes)
        
        return np.unique(train_genes)

        
class SurveyRegulatoryNetwork(CellOracleLinks):
    def __init__(self):
        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/survey/celloracle_links_spleen.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.cluster_labels = {
            '8': 'T',
            '4': 'Neutrophil',
            '5': 'Plasma_Cell',
            '0': 'B',
            '2': 'Macrophage',
            '3': 'NK',
            '6': 'Platelet',
            '7': 'RBC',
            '1': 'DC'
        }

        self.annot = 'cluster'



    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[str(label)]
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    


class DayThreeRegulatoryNetwork(CellOracleLinks):
    """
    CellOracle infered GRN 
    These are dataset specific and come with estimated betas and p-values
    """

    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/slideseq/celloracle_links_day3_1.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'rctd_cluster'

        with open(os.path.join(self.base_pth, 'celltype_assign.json'), 'r') as f:
            self.cluster_labels = json.load(f)


    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            # cluster = self.cluster_labels[str(label)]
            cluster = label
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    

class MouseKidneyRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/kidney/celloracle_links.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cluster'

        with open(os.path.join(self.base_pth, 'kidney/celltype_assign.json'), 'r') as f:
            self.cluster_labels = json.load(f)

    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            # cluster = self.cluster_labels[str(label)]
            cluster = label
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    


class HumanTonsilNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/BaseGRNs/tonsil_celloracle.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cluster'

        self.cluster_labels = {
            0: 'Plasma Cells',
            1: 'Cycling B Cells',
            2: 'Follicular Dendritic Cells ',
            3: 'Dark Zone B Cells',
            4: 'IFN B Cells',
            5: 'T Cells',
            6: 'Light Zone B Cells',
            7: 'Memory B Cells',
            8: 'Naive B Cells',
            9: 'GC-Tfh'
        }


    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[int(label)]
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    

class HumanTonsilRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/slidetags/tonsil_colinks.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cell_type_int'
        
        self.cluster_labels = {0: 'B_germinal_center',
            1: 'B_memory', 
            2: 'B_naive',
            3: 'FDC',
            4: 'NK',
            5: 'T_CD4',
            6: 'T_CD8',
            7: 'T_double_neg',
            8: 'T_follicular_helper',
            9: 'mDC',
            10: 'myeloid',
            11: 'pDC',
            12: 'plasma'
        }

    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[label]
            # cluster = str(label)
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators