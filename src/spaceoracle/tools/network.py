import celloracle as co
import numpy as np

class GeneRegulatoryNetwork:
    def __init__(self, organism='mouse'):
        if organism == 'mouse':
            self.data = co.data.load_mouse_scATAC_atlas_base_GRN()
            
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
                