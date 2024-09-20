import warnings
warnings.simplefilter(action='ignore', category=Warning)
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score
import torch
from pyro.infer import Predictive
from ..tools.utils import set_seed, seed_worker, deprecated
from ..tools.network import DayThreeRegulatoryNetwork
from ..tools.data import SpaceOracleDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_mean
from pyro.infer import SVI, Trace_ELBO
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
from joblib import Parallel, delayed

set_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

# device = torch.device('cpu')

class BaseEstimator(ABC):
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self):
        pass
    
    def _training_loop(self):
        raise NotImplementedError

    def _validation_loop(self):
        raise NotImplementedError

    def get_betas(self):
        raise NotImplementedError



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


class BayesianRegression(BaseEstimator):

    def __init__(self, n_regulators, device):
        self.linear_model = BayesianLinearLayer(n_regulators, 1, device=device)
        self.linear_model.to(device)
        self.n_regulators = n_regulators
        self.models_dict = {}
        self.guides = {}
        self.device = device
    
    def fit(self, X, y, cluster_labels, max_epochs=100, learning_rate=3e-2, num_samples=1000):
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
        results = Parallel(n_jobs=8)(delayed(fit_cluster)(cluster) for cluster in unique_clusters)

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
    
    # def simple_elbo(self, model, guide, *args, **kwargs):
    #     guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
    #     model_trace = pyro.poutine.trace(
    #         pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    #     return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())

    def _fit_one(self, X, y, max_epochs, learning_rate, num_samples):
        model = BayesianLinearLayer(self.n_regulators, 1, device=self.device)
        model.train()
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
        # guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": learning_rate, "weight_decay": 0.0})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        # svi = SVI(model, guide, adam, loss=self.simple_elbo)

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





# class BayesianEstimator(BaseEstimator):

#     def __init__(self, adata, target_gene, layer='imputed_count'):
#         assert target_gene in adata.var_names
#         assert layer in adata.layers

#         self.adata = adata
#         self.target_gene = target_gene
#         self.grn = DayThreeRegulatoryNetwork() 

#         self.regulators = self.grn.get_cluster_regulators(self.adata, self.target_gene)
#         self.n_clusters = len(self.adata.obs['rctd_cluster'].unique())
        
#         self.layer = layer
#         self.model = None
#         self.losses = []

#     @staticmethod
#     def _build_dataloaders_from_adata(adata, target_gene, regulators, batch_size=32, 
#     mode='train', rotate_maps=True, annot='rctd_cluster', layer='imputed_count', spatial_dim=64, test_size=0.2):

#         assert mode in ['train', 'train_test']
#         set_seed(42)
    
#         g = torch.Generator()
#         g.manual_seed(42)
        
#         params = {
#             'batch_size': batch_size,
#             'worker_init_fn': seed_worker,
#             'generator': g,
#             'pin_memory': False,
#             'num_workers': 0,
#             'drop_last': True,
#         }
        
#         dataset = SpaceOracleDataset(
#             adata.copy(), 
#             target_gene=target_gene, 
#             regulators=regulators, 
#             annot=annot, 
#             layer=layer,
#             spatial_dim=spatial_dim,
#             rotate_maps=rotate_maps
#         )

#         if mode == 'train':
#             train_dataloader = DataLoader(dataset, shuffle=True, **params)
#             valid_dataloader = DataLoader(dataset, shuffle=False, **params)
            
#             return train_dataloader, valid_dataloader
        
#         if mode == 'train_test':
#             split = int((1-test_size)*len(dataset))
#             generator = torch.Generator().manual_seed(42)
#             train_dataset, valid_dataset = random_split(
#                 dataset, [split, len(dataset)-split], generator=generator)
#             train_dataloader = DataLoader(train_dataset, shuffle=True, **params)
#             valid_dataloader = DataLoader(valid_dataset, shuffle=False, **params)

#             return train_dataloader, valid_dataloader
        
    
#     def predict_y(self, model, betas, inputs_x):

#         assert inputs_x.shape[1] == len(self.regulators) == model.output_dim
#         assert betas.shape[1] == model.output_dim+1, "intercept missing from betas"

#         y_pred = betas[:, 0]
#         for _w in range(model.output_dim):
#             y_pred += betas[:, _w+1]*inputs_x[:, _w]
#         return y_pred
        
    
#     def _training_loop(self, model, dataloader, svi):
#         model.train()
#         total_loss = 0

#         # The svi.step() method internally handles the forward pass, loss calculation,
#         # and backward pass (including loss.backward()), so we don't need to call
#         # loss.backward() explicitly here.
#         # ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]

#         for batch_spatial, batch_x, batch_y, batch_labels in dataloader:

#             loss = svi.step(
#                 batch_spatial.to(device), 
#                 batch_labels.to(device), 
#                 batch_x.to(device),
#                 y=batch_y.to(device)
#             )
#             total_loss += loss
        
#         return total_loss / len(dataloader)

#     @torch.no_grad()
#     def _validation_loop(self, model, dataloader, guide, num_samples=100, parallel=True, inference='svi'):
#         assert inference in ['svi', 'mc']
#         model.eval()
#         total_loss = 0


#         # The posterior distribution combines the prior knowledge with 
#         # information acquired from the data matrix X. We cannot directly 
#         # apply a Bayes rule to determine the posterior because the denominator 
#         # (the marginal distribution, integrated over the latent variables) 
#         # is intractable.

#         for batch_spatial, batch_x, batch_y, batch_labels in dataloader:

#             if inference == 'svi':

#                 # y_pred = model(
#                 #     batch_spatial.to(device), 
#                 #     batch_labels.to(device),
#                 #     batch_x.to(device),
#                 #     y=None
#                 # )

#                 predictive = Predictive(
#                     model, guide=guide, num_samples=num_samples, parallel=False,
#                     # return_sites=("linear.weight", "obs", "_RETURN")
#                     return_sites=("obs", "_RETURN")

#                 )
                
#                 samples = predictive(
#                     batch_spatial.to(device), 
#                     batch_labels.to(device), 
#                     batch_x.to(device),
#                     y=None
#                 )

#                 print(samples["_RETURN"].shape, samples["_RETURN"].mean(0).shape)

#                 y_pred = samples["_RETURN"].mean(0)
#                 # y_pred = samples["obs"].mean(0)


#             elif inference == 'mc':
#                 # In contrast to using variational inference which gives us an 
#                 # approximate posterior over our latent variables, we can also do 
#                 # exact inference using Markov Chain Monte Carlo (MCMC), a class of 
#                 # algorithms that in the limit, allow us to draw unbiased samples 
#                 # from the true posterior. The algorithm that we will be using is 
#                 # called the No-U Turn Sampler (NUTS) [1], which provides an efficient 
#                 # and automated way of running Hamiltonian Monte Carlo. It is slightly 
#                 # slower than variational inference, but provides an exact estimate.
#                 # see -> https://pyro.ai/examples/bayesian_regression_ii.html

#                 raise NotImplementedError


#             loss = ((y_pred - batch_y.to(device))**2).mean().item()
#             total_loss += loss
        
#         return total_loss / len(dataloader)
    

    
#     def fit(
#         self,
#         annot,
#         max_epochs=10, 
#         learning_rate=2e-4, 
#         spatial_dim=64,
#         batch_size=32, 
#         mode='train',
#         rotate_maps=True,
#         cluster_grn=False,
#         pbar=None
#         ):

#         assert annot in self.adata.obs.columns

#         self.spatial_dim = spatial_dim  
#         self.rotate_maps = rotate_maps
#         self.annot = annot
#         self.cluster_grn = cluster_grn

#         adata = self.adata

#         try:
#             model, losses = self._build_model(
#                 adata,
#                 annot,
#                 spatial_dim=spatial_dim, 
#                 mode=mode,
#                 layer=self.layer,
#                 cluster_grn=cluster_grn,
#                 max_epochs=max_epochs,
#                 batch_size=batch_size,
#                 learning_rate=learning_rate,
#                 rotate_maps=rotate_maps,
#                 pbar=pbar
#             )
            
#             self.model = model  
#             self.losses = losses
            
        
#         except KeyboardInterrupt:
#             print('Training interrupted...')
#             pass





#     def get_betas(self):
#         raise NotImplementedError
    
#     def export(self):
#         self.model.eval()
#         return self.model, self.regulators, self.target_gene