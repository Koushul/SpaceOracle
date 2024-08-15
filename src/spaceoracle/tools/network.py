# import celloracle as co
import numpy as np
import pandas as pd

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

# class GeneRegulatoryNetwork:
#     ## This GRN is specific to the day3_1 dataset
#     #TODO: make it more general

#     def __init__(self):
#         with open('../data/celloracle_links.pkl', 'rb') as f:
#             self.links_dict = pickle.load(f)

#     def get_regulators(self, adata, target_gene):
#         pass

                