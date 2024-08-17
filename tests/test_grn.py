import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import pickle
import pandas as pd
import numpy as np
import anndata
from spaceoracle.tools.network import GeneRegulatoryNetwork, DayThreeRegulatoryNetwork
import pickle


class TestGeneRegulatoryNetwork(unittest.TestCase):

    def setUp(self):
        # Mock the pickle files
        self.mock_links_day3_1 = pd.DataFrame({
            'source': ['gene1', 'gene2'],
            'target': ['gene3', 'gene4'],
            'importance': [0.5, 0.7]
        })
        self.mock_links_day3_2 = pd.DataFrame({
            'source': ['gene2', 'gene3'],
            'target': ['gene4', 'gene5'],
            'importance': [0.6, 0.8]
        })

        # Create a mock AnnData object
        n_obs = 100
        
        obs = {'cell_type': np.random.choice(['A', 'B', 'C'], size=n_obs)}
        var = {'gene_symbols': [
            'Pax5', 'Mef2c', 'Bcl11a', 'Ebf1', 'Pou2f2',
            'Irf9', 'Tbp', 'Stat2', 'Tal1', 'Myc',
            'Rreb1', 'Ebf3', 'Rxra', 'Foxj3', 'Cd74'
        ]}
        X = np.random.rand(n_obs, len(var['gene_symbols']))
        self.mock_adata = anndata.AnnData(X=X, obs=obs, var=var)
        self.mock_adata.var_names = self.mock_adata.var['gene_symbols']

    def test_init(self):
        grn = GeneRegulatoryNetwork()
        self.assertIsInstance(grn, GeneRegulatoryNetwork)

    def test_get_regulators(self):
        grn = GeneRegulatoryNetwork()
        target_gene = 'Cd74'
        regulators = grn.get_regulators(self.mock_adata, target_gene)

        self.assertListEqual(
            regulators, 
            [   'Bcl11a', 'Ebf1', 'Ebf3', 
                'Foxj3', 'Irf9', 'Mef2c', 
                'Myc', 'Pax5', 'Pou2f2', 
                'Rreb1', 'Rxra', 'Stat2', 
                'Tal1', 'Tbp'
            ])

    def test_celloracle_links(self):
        grn = DayThreeRegulatoryNetwork()
        target_gene = 'Cd74'
        regulators = grn.get_regulators(self.mock_adata, target_gene)
        
        self.assertListEqual(
            regulators, 
            [
                'Atf3', 'Bcl11a', 'Ebf1', 'Ebf3', 
                'Egr1', 'Egr2', 'Foxj3', 'Gata2', 
                'Hivep1', 'Irf9', 'Klf7', 'Mef2a', 
                'Mef2c', 'Myc', 'Pax5', 'Pou2f2', 
                'Rreb1', 'Rxra', 'Stat2', 'Taf1', 
                'Tal1', 'Tbp', 'Tcf7l2', 'Zkscan1'
            ])

if __name__ == '__main__':
    unittest.main()
