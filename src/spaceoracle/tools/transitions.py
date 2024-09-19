import scanpy as sc 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


class COTransitions:
    def get_embed_spaces(so):
        sc.tl.pca(so.adata, svd_solver='arpack')
        sc.pp.neighbors(so.adata)
        sc.tl.umap(so.adata)

    def estimate_transitions(so, embedding='X_spatial', test_co=False, n_grid=40, n_neighbors=200):
        so.estimate_transition_prob(embedding_name=embedding, test_co=test_co)
        so.calculate_embedding_shift(sigma_corr=0.05)
        so.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=n_neighbors)

    def plot_quiver(scale=25, goi='goi'):
        fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

        scale = scale
        # Show quiver plot
        so.plot_quiver(scale=scale, ax=ax[0])
        ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

        # Show quiver plot that was calculated with randomized graph.
        so.plot_quiver_random(scale=scale, ax=ax[1])
        ax[1].set_title(f"Randomized simulation vector")

        plt.show()
    
    def plot_simulated(scale_simulation, min_mass, plot=False):
        so.calculate_mass_filter(min_mass=min_mass, plot=plot)

        fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

        # Show quiver plot
        so.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
        ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

        # Show quiver plot that was calculated with randomized graph.
        so.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
        ax[1].set_title(f"Randomized simulation vector")

        plt.show()

