from numba import jit
import numpy as np
import torch
import random
import functools
import inspect
import warnings
import pickle
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy import sparse
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def knn_distance_matrix(data, metric=None, k=40, mode='connectivity', n_jobs=4):
    """Calculate a nearest neighbour distance matrix

    Notice that k is meant as the actual number of neighbors NOT INCLUDING itself
    To achieve that we call kneighbors_graph with X = None
    """
    if metric == "correlation":
        nn = NearestNeighbors(
            n_neighbors=k, metric="correlation", 
            algorithm="brute", n_jobs=n_jobs)
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)
    else:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)

def connectivity_to_weights(mknn, axis=1):
    if type(mknn) is not sparse.csr_matrix:
        mknn = mknn.tocsr()
    return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))

def convolve_by_sparse_weights(data, w):
    w_ = w.T
    assert np.allclose(w_.sum(0), 1)
    return sparse.csr_matrix.dot(data, w_)


def _adata_to_matrix(adata, layer_name, transpose=True):
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


class DeprecatedWarning(UserWarning):
    pass

def deprecated(instructions=''):
    """Flags a method as deprecated.

    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = '{} is a deprecated function. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(message,
                                   category=DeprecatedWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clean_up_adata(adata, fields_to_keep):
    current_obs_fields = adata.obs.columns.tolist()
    excess_obs_fields = [field for field in current_obs_fields if field not in fields_to_keep]
    for field in excess_obs_fields:
        del adata.obs[field]
    
    current_var_fields = adata.var.columns.tolist()
    excess_var_fields = [field for field in current_var_fields 
        if field not in []]
    for field in excess_var_fields:
        del adata.var[field]

    del adata.uns


@jit
def gaussian_kernel_2d(origin, xy_array, radius, eps=0.001):
    distances = np.sqrt(np.sum((xy_array - origin)**2, axis=1))
    sigma = radius / np.sqrt(-2 * np.log(eps))
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    # weights[0] = 0
    return weights