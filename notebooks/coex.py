import sys
sys.path.append('../src')

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from utils import get_immune_genes, process_adata, get_markers

immune_genes = get_immune_genes(mouse=True)
adata = sc.read_h5ad('../data/slideseq/day3_1.h5ad')
adata = process_adata(
    adata, n_top_genes=5000, include_genes=immune_genes, mouse=True)
print(adata.shape)

clusters = list(adata.obs['rctd_cluster'].value_counts().keys())


coex_dir = '../data/slideseq/coexpression/'
adata_copy = adata.copy()



if __name__ == '__main__':
    from arboreto.algo import grnboost2
    from distributed import LocalCluster, Client

    # Initialize local Dask cluster with 1 worker and 1 thread
    local_cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='40GB')
    custom_client = Client(local_cluster)

    for cluster in clusters:
        print(f'Starting cluster {cluster}')
        
        # Subset adata based on the cluster
        cluster_adata = adata_copy[adata_copy.obs['rctd_cluster'] == cluster]

        # Convert the expression matrix to a DataFrame
        ex_matrix = cluster_adata.to_df()

        # Use gene names from the cluster-specific adata
        tf_names = list(cluster_adata.var_names)

        # Run GRNBoost2 to infer gene regulatory networks
        network = grnboost2(expression_data=ex_matrix,
                            tf_names=tf_names,
                            client_or_address=custom_client,
                            verbose=True)

        # Save the inferred network to a CSV file
        network.to_csv(f'../data/slideseq/coexpression/grnboost2_c{cluster}.tsv', sep='\t', index=False, header=False)

    