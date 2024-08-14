import wandb
import numpy as np
import random
import sys
import anndata
import scanpy as sc

from ..models.estimators import GeoCNNEstimatorV2

# If import issues, run from spaceoracle directory:
# python -m spaceoracle.tools.sweep_cnn

sweep_config = {
    "name": "sweep",
    "method": "random",
    "metric": {
        "goal": "minimize", 
        "name": "validation_loss"},
    "parameters": {
        "epochs": {"values": [15, 30, 50, 100, 150]},
        "lr": {"min": 0.0001, "max": 0.1},
        "spatial_dim": {"values": [32, 64, 128, 256]},
        "batch_size": {"values": [16, 32, 64]},
        "beta_init": {"values": ['ols', 'ones']},
        "rotate_maps": {"values":[True, False]},
        "regularize": {"values":[True, False]},
        "lambd": {"min": 0.0001, "max": 0.5},
        "a": {"min": 0.0, "max": 1.0}
    },
}

def start_sweep():
    adata_train, adata_test = ugly_adata_preprocess()
    estimator = GeoCNNEstimatorV2(adata_train, target_gene='Cd74')

    wandb.init()
    config = wandb.config

    estimator.fit(
        annot='rctd_cluster', 
        max_epochs=config.epochs, 
        learning_rate=config.lr, 
        spatial_dim=config.spatial_dim,
        batch_size=config.batch_size,
        init_betas=config.beta_init,
        mode='train_test',
        rotate_maps=config.rotate_maps,
        regularize=config.regularize,
        lambd=config.lambd, a=config.a,
        use_wandb=True
    )

def ugly_adata_preprocess():
    adata_train = anndata.read_h5ad('../data/slideseq/day3_1.h5ad')
    adata_test = anndata.read_h5ad('../data/slideseq/day3_2.h5ad')

    n_top_genes = 4000
    min_cells = 10
    min_counts = 350

    ### Prepare train ###

    adata_train.var_names_make_unique()
    adata_train.var["mt"] = adata_train.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata_train, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata_train, min_counts=min_counts)
    adata_train = adata_train[adata_train.obs["pct_counts_mt"] < 20].copy()
    adata_train = adata_train[:, ~adata_train.var["mt"]]
    sc.pp.filter_genes(adata_train, min_cells=min_cells)

    adata_train.layers["raw_count"] = adata_train.X

    sc.pp.normalize_total(adata_train, inplace=True)
    sc.pp.log1p(adata_train)
    sc.pp.highly_variable_genes(adata_train, flavor="seurat", n_top_genes=n_top_genes)

    adata_train = adata_train[:, adata_train.var.highly_variable]

    ### Prepare test ###
    
    adata_test.var_names_make_unique()
    adata_test.var["mt"] = adata_test.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata_test, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata_test, min_counts=min_counts)
    adata_test = adata_test[adata_test.obs["pct_counts_mt"] < 20].copy()
    adata_test = adata_test[:, ~adata_test.var["mt"]]
    sc.pp.filter_genes(adata_test, min_cells=min_cells)

    adata_test.layers["raw_count"] = adata_test.X

    sc.pp.normalize_total(adata_test, inplace=True)
    sc.pp.log1p(adata_test)
    sc.pp.highly_variable_genes(
        adata_test, flavor="seurat", n_top_genes=n_top_genes)

    adata_test = adata_test[:, adata_test.var.highly_variable]

    ### Take joint ###

    np.intersect1d(adata_train.var_names, adata_test.var_names).__len__()
    adata_train = adata_train[:, adata_train.var_names.isin(np.intersect1d(adata_train.var_names, adata_test.var_names))]
    adata_test = adata_test[:, adata_test.var_names.isin(np.intersect1d(adata_train.var_names, adata_test.var_names))]

    return adata_train, adata_test


if __name__ == '__main__':
    
    # sweep_id = wandb.sweep(sweep=sweep_config, project="SpaceOracle")
    # wandb.agent(sweep_id, function=start_sweep, count=5)
    
    wandb.agent("alw399/SpaceOracle/q9z17eqv", function=start_sweep, count=5)