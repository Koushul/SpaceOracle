import wandb
import numpy as np
import random
import sys
import anndata
import scanpy as sc
import torch
from scipy.stats import pearsonr

from ..models.estimators import ViTEstimatorV2
from ..callbacks.fixtures.simulator import SimulatedDataV2
from ..callbacks.simulation_callback import SimulationBetaCallbackV2

# If import issues, run from src directory:
# python -m spaceoracle.tools.sweep_vit

sweep_config = {
    "name": "vitsweep",
    "method": "random",
    "metric": {
        "goal": "minimize", 
        "name": "rdata_val"},
    "parameters": {
        "epochs": {"values": [50, 100, 150]},
        "lr": {"min": 0.0001, "max": 0.005},
        "spatial_dim": {"values": [32, 64, 128, 256]},
        "batch_size": {"values": [16, 32, 64]},
        "beta_init": {"values": ['ols', 'ones']},
        "rotate_maps": {"values":[True, False]},
        "regularize": {"values":[True, False]},
        "n_patches": {"values": [2, 4, 8, 16, 32, 64]},
        "n_heads": {"values": [1, 2, 4, 8, 16]},
        "n_blocks": {"values": [1, 2, 3, 4]},
        "hidden_d": {"values": [8, 16, 32, 64]},
        "lambd": {"min": 0.0001, "max": 0.5},
        "a": {"min": 0.0, "max": 1.0}
    },
}

def start_sweep():

    wandb.init()
    config = wandb.config

    ### Train/evaluate on simulated data
    print('Using simulated data')

    sim_data = SimulatedDataV2(density=20, ntfs=10, clusters=6, position='wave')
    estimator = ViTEstimatorV2(sim_data.adata, target_gene='target_gene', 
                        regulators=sim_data.tf_labels, n_clusters=sim_data.clusters)

    estimator.fit(
        annot='sim_cluster', 
        max_epochs=config.epochs, 
        learning_rate=config.lr, 
        spatial_dim=config.spatial_dim,
        batch_size=config.batch_size,
        init_betas=config.beta_init,
        mode='train_test',
        rotate_maps=config.rotate_maps,
        regularize=config.regularize,
        n_patches=config.n_patches, 
        n_heads=config.n_heads, 
        n_blocks=config.n_blocks, 
        hidden_d=config.hidden_d,
        lambd=config.lambd, a=config.a,
        use_wandb='simulated'
    )

    record_pearson(sim_data, estimator)

    ### Train/evaluate on real data
    print('Evaluating real data')

    adata_train, _ = ugly_adata_preprocess()
    estimator = ViTEstimatorV2(adata_train, target_gene='Cd74')

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
        n_patches=config.n_patches, 
        n_heads=config.n_heads, 
        n_blocks=config.n_blocks, 
        hidden_d=config.hidden_d,
        lambd=config.lambd, a=config.a,
        use_wandb='rdata'
    )


def record_pearson(sim_data, estimator):
    y = sim_data.y
    beta_pred, y_pred = get_preds(sim_data, estimator)
    
    check_betas = SimulationBetaCallbackV2()
    beta_pearson = check_betas(beta_pred, sim_data.betas)

    corr, _ = pearsonr(y.reshape(-1), y_pred.reshape(-1))

    table = wandb.Table(
        columns=list(beta_pearson.keys()), 
        data=[[float(value) for value in beta_pearson.values()]]  # Convert to list of floats
    )

    wandb.log({'beta_pearson': table, 'simulated_ycorr': corr})

def get_preds(sim_data, estimator):
    with torch.no_grad():
        beta_pred = estimator.get_betas(
            np.array(sim_data.adata.obsm['spatial']),
            np.array(sim_data.adata.obs['sim_cluster']),
        )
        y_pred = estimator.predict_y(
            estimator.model, 
            torch.from_numpy(beta_pred).cuda(), 
            torch.from_numpy(sim_data.adata.to_df()[estimator.regulators].values).cuda()).cpu().numpy()
    return beta_pred, y_pred


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
    # sweep_id = wandb.sweep(sweep=sweep_config, project="vit")
    # wandb.agent(sweep_id, function=start_sweep, count=2)

    wandb.agent("SpaceTeam/vit/08yc2k6v", function=start_sweep, count=40)