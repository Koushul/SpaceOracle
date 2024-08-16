# import celloracle as co
import numpy as np
import pandas as pd
import pickle
import os

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
        
        tf = base_GRN[base_GRN.gene_short_name==target_gene][
                np.intersect1d(
                    adata.var_names, 
                    base_GRN[base_GRN.gene_short_name==target_gene].columns
                )
            ].sum()
        
        tf = tf[tf!=0]
        
        return tf.index.tolist()


class CellOracleLinks:
    
    def __init__(self):
        pass

    def get_regulators(self, adata, target_gene):
        pass


class DayThreeRegulatoryNetwork(CellOracleLinks):
    """
    CellOracle infered GRN 
    These are dataset specific and come with estimated betas and p-values
    """

    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data', 'slideseq')

        with open(self.base_pth+'/celloracle_links_day3_1.pkl', 'rb') as f:
            self.links_day3_1 = pickle.load(f)

        # with open(self.base_pth+'/celloracle_links_day3_2.pkl', 'rb') as f:
        #     self.links_day3_2 = pickle.load(f)

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
                for link_data in self.links_day3_1.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'source.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    def get_targets_with_pvalues(self, adata, tf, alpha=0.05):
        assert tf in adata.var_names, f'{tf} not in adata.var_names'
        co_links = pd.concat(
            [link_data.query(f'source == "{tf}" and p < {alpha}')[['target', 'coef_mean']] 
                for link_data in self.links_day3_1.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'target.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
                