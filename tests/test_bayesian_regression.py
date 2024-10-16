import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import torch
import numpy as np
from spaceoracle.models.probabilistic_estimators import BayesianRegression
from sklearn.datasets import make_regression

@pytest.fixture
def bayesian_regression():
    device = torch.device("cpu")
    return BayesianRegression(n_regulators=5, device=device)

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    cluster_labels = np.random.randint(0, 3, size=100)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32), torch.tensor(cluster_labels)

# Add a new fixture to store timing information
@pytest.fixture(scope="module")
def timing_info():
    return {}

# Modify each test function to measure and store execution time
@pytest.mark.usefixtures("timing_info")
def test_initialization(bayesian_regression, timing_info):
    start = time.time()
    assert bayesian_regression.n_regulators == 5
    assert isinstance(bayesian_regression.linear_model, torch.nn.Module)
    assert len(bayesian_regression.models_dict) == 0
    assert len(bayesian_regression.guides) == 0
    end = time.time()
    timing_info['test_initialization'] = end - start

@pytest.mark.usefixtures("timing_info")
def test_fit_sequential(bayesian_regression, sample_data, timing_info):
    start = time.time()
    X, y, cluster_labels = sample_data
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100, parallel=False)
    end = time.time()
    print(f'Time taken: {end-start} seconds')
    assert len(bayesian_regression.models_dict) == 3
    assert len(bayesian_regression.guides) == 3
    timing_info['test_fit_sequential'] = end - start
    
    for cluster in range(3):
        assert cluster in bayesian_regression.models_dict
        assert cluster in bayesian_regression.guides

@pytest.mark.usefixtures("timing_info")
def test_fit_parallel(bayesian_regression, sample_data, timing_info):
    start = time.time()
    X, y, cluster_labels = sample_data
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100, parallel=True)
    
    assert len(bayesian_regression.models_dict) == 3
    assert len(bayesian_regression.guides) == 3
    end = time.time()
    timing_info['test_fit_parallel'] = end - start
    for cluster in range(3):
        assert cluster in bayesian_regression.models_dict
        assert cluster in bayesian_regression.guides

@pytest.mark.usefixtures("timing_info")
def test_get_betas(bayesian_regression, sample_data, timing_info):
    X, y, cluster_labels = sample_data
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100)
    
    cluster = 0
    betas = bayesian_regression.get_betas(X[cluster_labels == cluster], cluster, num_samples=100)
    
    assert isinstance(betas, np.ndarray)
    assert betas.shape[1] == bayesian_regression.n_regulators + 1  # n_regulators + intercept
    timing_info['test_get_betas'] = time.time() - time.time()

@pytest.mark.usefixtures("timing_info")
def test_score(bayesian_regression, sample_data, timing_info):
    X, y, cluster_labels = sample_data
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100)
    
    cluster = 0
    X_test = X[cluster_labels == cluster]
    y_test = y[cluster_labels == cluster]
    
    model = bayesian_regression.models_dict[cluster]
    guide = bayesian_regression.guides[cluster]
    
    score = bayesian_regression._score(model, guide, X_test, y_test, num_samples=100)
    
    assert isinstance(score, float)
    assert -1 <= score <= 1  # R2 score range
    timing_info['test_score'] = time.time() - time.time()

@pytest.mark.usefixtures("timing_info")
def test_fit_one(bayesian_regression, sample_data, timing_info):
    X, y, _ = sample_data
    model, guide = bayesian_regression._fit_one(X, y, max_epochs=10, 
        learning_rate=1e-2, num_samples=100)
    
    assert isinstance(model, torch.nn.Module)
    assert callable(guide)
    timing_info['test_fit_one'] = time.time() - time.time()

@pytest.mark.usefixtures("timing_info")
def test_different_cluster_sizes(bayesian_regression, timing_info):
    X = torch.randn(100, 5)
    y = torch.randn(100)
    cluster_labels = torch.tensor([0] * 60 + [1] * 30 + [2] * 10)
    
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100)
    
    assert len(bayesian_regression.models_dict) == 3
    assert len(bayesian_regression.guides) == 3
    timing_info['test_different_cluster_sizes'] = time.time() - time.time()

@pytest.mark.usefixtures("timing_info")
def test_get_betas_shape(bayesian_regression, sample_data, timing_info):
    X, y, cluster_labels = sample_data
    bayesian_regression.fit(X, y, cluster_labels, max_epochs=10, 
        learning_rate=1e-2, num_samples=100)
    
    for cluster in range(3):
        X_cluster = X[cluster_labels == cluster]
        betas = bayesian_regression.get_betas(X_cluster, cluster, num_samples=100)
        assert betas.shape == (100, bayesian_regression.n_regulators + 1)
        timing_info['test_get_betas_shape'] = time.time() - time.time()

# Add a new function to print the timing table
@pytest.mark.usefixtures("timing_info")
def test_print_timing_table(timing_info):
    print("\nFunction Execution Times:")
    print("-------------------------")
    print("{:<30} {:<10}".format("Function Name", "Time (s)"))
    print("-------------------------")
    for func_name, exec_time in timing_info.items():
        print("{:<30} {:.4f}".format(func_name, exec_time))
    print("-------------------------")

# Make sure this is the last test function
def test_z_print_timing_table(timing_info):
    test_print_timing_table(timing_info)


