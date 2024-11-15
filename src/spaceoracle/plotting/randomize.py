import numpy as np 
from numba import jit

from .transitions import estimate_transitions_2D, estimate_transitions_3D, view_probabilities
from .shift import estimate_transitions


def randomize_transitions(adata, delta_X, embedding, annot='rctd_cluster', n_neighbors=200, vector_scale=1,
    visual_clusters=['B-cell', 'Th2', 'Cd8 T-cell'], renormalize=False, n_jobs=1):

    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    estimate_transitions(adata, delta_X_rndm , embedding, annot, n_neighbors,
        vector_scale, visual_clusters, renormalize, n_jobs)
    
def randomize_view_probabilities(adata, delta_X, embedding, cluster=None, annot=None, log_P=True, n_jobs=1):
    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    view_probabilities(adata, delta_X_rndm, embedding, cluster, annot, log_P, n_jobs)



def randomize_transitions_2D(adata, delta_X, embedding, annot=None, normalize=True, 
n_neighbors=200, vector_scale=0.1, grid_scale=1, n_jobs=1):

    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    estimate_transitions_2D(adata, delta_X_rndm , embedding, annot, normalize, 
        n_neighbors, vector_scale, grid_scale, n_jobs)


def randomize_transitions_3D(adata, delta_X, embedding, annot=None, normalize=True, 
vector_scale=0.1, grid_scale=1, n_jobs=1):

    delta_X_rndm = np.copy(delta_X)
    permute_rows_nsign(delta_X_rndm)

    estimate_transitions_3D(adata, delta_X_rndm, embedding, annot, normalize, 
        vector_scale, grid_scale, n_jobs)


# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
