import numpy as np 
from numba import jit

from .transitions import estimate_transitions_2D, estimate_transitions_3D


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
