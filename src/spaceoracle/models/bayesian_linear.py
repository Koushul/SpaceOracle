import numpy as np
import copy
from sklearn.metrics import r2_score
import torch
from pyro.infer import Predictive
import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.infer import SVI, Trace_ELBO
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
from joblib import Parallel, delayed
from spaceoracle.models.estimators import AbstractEstimator
import os

available_cores = os.cpu_count()


class BayesianLinearLayer(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, device=torch.device('cpu')):
        super().__init__()

        #  In order to make our linear regression Bayesian, 
        #  we need to put priors on the parameters weight and bias from nn.Linear. 
        #  These are distributions that represent our prior belief about 
        #  reasonable values for and (before observing any data).

        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.out_features = out_features
        self.in_features = in_features
        self.device = device

        self.linear.weight = PyroSample(
            prior=dist.Normal(
                torch.tensor(0., device=self.device), 0.1).expand(
                    [out_features, in_features]).to_event(2))
        
        self.linear.bias = PyroSample(
            prior=dist.Normal(
                torch.tensor(0., device=self.device), 0.1).expand(
                    [out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample(
            "sigma",
            dist.LogNormal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device)
            )
        )

        mean = self.linear(x).squeeze(-1)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class BayesianRegression(AbstractEstimator):

    def __init__(self, n_regulators, device):
        self.linear_model = BayesianLinearLayer(n_regulators, 1, device=device)
        self.linear_model.to(device)
        self.n_regulators = n_regulators
        self.models_dict = {}
        self.guides = {}
        self.device = device

    
    def fit(self, X, y, cluster_labels, max_epochs=100, learning_rate=3e-2, num_samples=1000, parallel=True):
        """
        In order to do inference, i.e. learn the posterior distribution over our 
        unobserved parameters, we will use Stochastic Variational Inference (SVI). 
        The guide determines a family of distributions, and SVI aims to find an 
        approximate posterior distribution from this family that has the lowest KL 
        divergence from the true posterior.
        """

        assert len(X) == len(y) == len(cluster_labels)

        def fit_cluster(cluster):
            _X = X[cluster_labels == cluster]
            _y = y[cluster_labels == cluster]
            # print(f'Cluster {cluster+1}/{len(np.unique(cluster_labels))} |> N={len(_X)}')
            model, guide = self._fit_one(_X, _y, max_epochs, learning_rate, num_samples)
            return cluster, model, guide

        unique_clusters = np.unique(cluster_labels)

        if parallel:
            n_jobs = min(available_cores-1, len(unique_clusters))
            print(f'Fitting {len(unique_clusters)} models in parallel... with {n_jobs}/{available_cores} cores')
            results = Parallel(n_jobs=n_jobs)(delayed(fit_cluster)(cluster) for cluster in unique_clusters)
        else:
            results = [fit_cluster(cluster) for cluster in tqdm(
                unique_clusters, desc='Fitting models sequentially...')]

        for cluster, model, guide in results:
            self.models_dict[cluster] = model
            self.guides[cluster] = guide


    def _score(self, model, guide, X_test, y_test, num_samples=1000):
        ## note: sampling from the posterior is expensive
        predictive = Predictive(
            model, guide=guide, num_samples=num_samples, parallel=False,
            return_sites=("obs", "_RETURN")
        )
        samples = predictive(X_test.to(self.device))
        y_pred = samples['obs'].mean(0).detach().cpu().numpy()

        return r2_score(y_test.cpu().numpy(), y_pred)


    def _fit_one(self, X, y, max_epochs, learning_rate, num_samples):
        model = BayesianLinearLayer(self.n_regulators, 1, device=self.device)
        model.train()
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
        # guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": learning_rate, "weight_decay": 0.0})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        pyro.clear_param_store()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # The svi.step() method internally handles the forward pass, loss calculation,
        # and backward pass (including loss.backward()), so we don't need to call
        # loss.backward() explicitly here.
        # ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]

        best_model = copy.deepcopy(model)
        best_score = -np.inf

        with tqdm(range(max_epochs), disable=True) as pbar:
            for epoch in pbar:
                loss = svi.step(
                    X_train.to(self.device), 
                    y_train.to(self.device)
                ) / y_train.numel()

                
                if (epoch==0 or epoch > 0.25*max_epochs) and \
                      epoch % int(max_epochs/10) == 0:
                    
                    r2 = self._score(model, guide, X_test, y_test, num_samples=num_samples)
                    if r2 <= best_score:
                        break
                    else:
                        best_model = copy.deepcopy(model)
                        best_score = r2
                    pbar.set_description(f"R2: {r2:.3f}")

        best_model.eval()
        return best_model, guide



    def get_betas(self, X, cluster, num_samples=1000):
        pyro.clear_param_store()
        model = self.models_dict[cluster]
        guide = self.guides[cluster]

        predictive = Predictive(
            model, guide=guide, num_samples=num_samples, parallel=False,
            return_sites=("linear.bias", "linear.weight", "obs", "_RETURN")
        )
        samples = predictive(X.to(self.device))

        beta_0 = samples['linear.bias'].view(-1, 1)
        betas = samples['linear.weight'].view(-1, self.n_regulators)

        return torch.cat([beta_0, betas], dim=1).detach().cpu().numpy()

