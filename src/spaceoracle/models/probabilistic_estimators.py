from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import copy
import enlighten
import pyro
from pyro.infer import SVI, Trace_ELBO
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
import os
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, random_split


from joblib import Parallel, delayed

from spaceoracle.models.spatial_map import xyc2spatial_fast
from spaceoracle.models.estimators import VisionEstimator, AbstractEstimator
from .pixel_attention import NicheAttentionNetwork
from ..tools.utils import set_seed, seed_worker
from ..tools.data import LigRecDataset


set_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

available_cores = os.cpu_count()

pyro.clear_param_store()


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
            results = [fit_cluster(cluster) for cluster in tqdm(unique_clusters, desc='Fitting models sequentially...')]

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






class ProbabilisticPixelAttention(VisionEstimator):

    def _build_model(
        self,
        adata,
        annot,
        spatial_dim,
        mode,
        layer,
        max_epochs,
        batch_size,
        learning_rate,
        rotate_maps,
        pbar=None
        ):

        train_dataloader, valid_dataloader = self._build_dataloaders_from_adata(
            adata, self.target_gene, self.regulators, 
            mode=mode, rotate_maps=rotate_maps, 
            batch_size=batch_size, annot=annot, 
            layer=layer,
            spatial_dim=spatial_dim
        )

        model = NicheAttentionNetwork(
            n_regulators=len(self.regulators),
            in_channels=self.n_clusters,
            spatial_dim=spatial_dim,
        )


        model.to(device)

        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
    
        # baseline_loss = self._estimate_baseline(valid_dataloader, self.beta_init)
        _prefix = f'[{self.target_gene} / {len(self.regulators)}]'

        if pbar is None:
            _manager = enlighten.get_manager()
            pbar = _manager.counter(
                total=max_epochs, 
                desc=f'{_prefix} <> MSE: ...', 
                unit='epochs'
            )
            pbar.refresh()

        for epoch in range(max_epochs):
            training_loss = self._training_loop(
                model, train_dataloader, criterion, optimizer)
            validation_loss = self._validation_loop(
                model, valid_dataloader, criterion)
            
            losses.append(validation_loss)

            if validation_loss < best_score:
                best_score = validation_loss
                best_model = copy.deepcopy(model)
                best_iter = epoch
            
            pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4g}'
            pbar.update()
            
        best_model.eval()
        
        return best_model, losses
    
    def predict_y(self, model, betas, batch_labels, inputs_x, anchors=None):

        assert inputs_x.shape[1] == len(self.regulators) == model.dim-1
        assert betas.shape[1] == len(self.regulators)+1

        if anchors is None:
            anchors = np.stack(
                [self.beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
                axis=0
            )
        
        anchors = torch.from_numpy(anchors).float().to(device)

        y_pred = anchors[:, 0]*betas[:, 0]
         
        for w in range(model.dim-1):
            y_pred += anchors[:, w+1]*betas[:, w+1]*inputs_x[:, w]

        return y_pred
    
    
    def _training_loop(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0

        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            
            optimizer.zero_grad()
            betas = model(batch_spatial.to(device), batch_labels.to(device))

            anchors = np.stack(
                [self.beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
                axis=0
            )
            
            outputs = self.predict_y(model, betas, batch_labels, inputs_x=batch_x.to(device), anchors=anchors)

            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            loss += 1e-3*((betas.mean(0) - torch.from_numpy(anchors).float().mean(0).to(device))**2).sum()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)
    

    # To test if values are significant in a Bayesian model, we can use the posterior distributions of the parameters.
    # A common approach is to compute the credible intervals (CIs) for the parameters of interest.
    # If the credible interval does not include zero, we can consider the effect to be significant.
    # So a marginal posterior distribution for a given IV that does not include 0 in the 95% HDI 
    # just shows that 95% of the most likely parameter values based on the data do not include zero. 
    
    def test_significance(self, betas, alpha=0.05):
        lower_bound = np.percentile(betas, 100 * (alpha / 2), axis=0)
        upper_bound = np.percentile(betas, 100 * (1 - alpha / 2), axis=0)
        significant = (lower_bound > 0) | (upper_bound < 0)
        
        return significant



    def fit(
        self,
        annot,
        max_epochs=10, 
        learning_rate=2e-4, 
        spatial_dim=64,
        batch_size=32, 
        alpha=0.05,
        num_samples=1000,
        mode='train_test',
        rotate_maps=True,
        parallel=True,
        cache=False,
        pbar=None
        ):
        
        assert annot in self.adata.obs.columns

        self.spatial_dim = spatial_dim  
        self.rotate_maps = rotate_maps
        self.annot = annot

        adata = self.adata
        beta_dists_file = f"{self.target_gene}_beta_dists.pkl"

            
        X = torch.from_numpy(adata.to_df(layer=self.layer)[self.regulators].values).float()
        y = torch.from_numpy(adata.to_df(layer=self.layer)[self.target_gene].values).float()
        cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long()
        if not cache:

            self.beta_model = BayesianRegression(
                n_regulators=len(self.regulators), device=torch.device('cpu'))

            self.beta_model.fit(
                X, y, cluster_labels, 
                max_epochs=3000, learning_rate=3e-3, 
                num_samples=num_samples,
                parallel=parallel
            )

            self.beta_dists = {}
            for cluster in range(self.n_clusters):
                self.beta_dists[cluster] = self.beta_model.get_betas(
                    X[cluster_labels==cluster].to(self.beta_model.device), 
                    cluster=cluster, 
                    num_samples=1000
                )
        
            with open(beta_dists_file, 'wb') as f:
                pickle.dump(self.beta_dists, f)

        else:
            with open(beta_dists_file, 'rb') as f:
                self.beta_dists = pickle.load(f)

        self.is_real = pd.DataFrame(
            [self.test_significance(self.beta_dists[i][:, 1:], alpha=alpha) for i in self.beta_dists.keys()], 
            columns=self.regulators
        ).T

        for c in self.is_real.columns:
            for ix, s in enumerate(self.is_real[c].values):
                if not s:
                    self.beta_dists[c][:, ix+1] = 0


        del X, y, cluster_labels

        try:
            model, losses = self._build_model(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                layer=self.layer,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rotate_maps=rotate_maps,
                pbar=pbar
            )
            
            self.model = model  
            self.losses = losses
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')


    def export(self):
        self.model.eval()

        return (
            self.model, 
            self.beta_dists, 
            self.is_real, 
            self.regulators, 
            self.target_gene
        )
    
            
    @torch.no_grad()
    def get_betas(self, xy=None, spatial_maps=None, labels=None, spatial_dim=None, beta_dists=None, layer=None):

        assert xy is not None or spatial_maps is not None
        assert beta_dists is not None or self.beta_dists is not None



        spatial_dim = self.spatial_dim if spatial_dim is None else spatial_dim
        
        if spatial_maps is None:
            spatial_maps = xyc2spatial_fast(
                xyc = np.column_stack([xy, labels]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            ).astype(np.float32)
            
        
        spatial_maps = torch.from_numpy(spatial_maps)

        dataset = TensorDataset(
            spatial_maps.float(), 
            torch.from_numpy(labels).long()
        )   

        g = torch.Generator()
        g.manual_seed(42)
        
        params = {
            'batch_size': 1024,
            'worker_init_fn': seed_worker,
            'generator': g
        }
        
        infer_dataloader = DataLoader(dataset, shuffle=False, **params)

        beta_list = []
            
        for batch_spatial, batch_labels in infer_dataloader:
            betas = self.model(batch_spatial.to(device), batch_labels.to(device))
            beta_list.extend(betas.cpu().numpy())
        
        return np.array(beta_list)


class ProbabilisticPixelAttentionLR(ProbabilisticPixelAttention):
    def __init__(self, adata, target_gene, grn=None, lrn=None, regulators=None, layer='imputed_count'):
        super().__init__(adata, target_gene, grn=grn, regulators=regulators, layer=layer)
        self.ligrec_net = lrn
        self.ligands = self.ligrec_net.gl_dict.get(target_gene, [])
        self.ligand_affected = len(self.ligands)
        if self.ligand_affected:
            self.receptor_beta_dicts = self.build_lr_model(self.ligands, lrn.lr_dict)
            self.receptors = sorted(self.receptor_beta_dicts.keys()) # ensure same order as dataloader

    def build_lr_model(self, ligands, lr_dict, embed_dim=5):
        def get_receptor_beta_model(receptor_ligands):
            return nn.Sequential(
                        nn.Linear(len(receptor_ligands), embed_dim), # batch, distance, expr for each cell within radius
                        nn.ReLU(),
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(embed_dim)
                        ).to(device)
        
        # reverse dictionary, get the ligands that affect each receptor
        rl_dict = defaultdict(list)
        for idx, ligand in enumerate(ligands):
            receptors = lr_dict[ligand]
            for receptor in receptors:
                if ligand not in rl_dict[receptor]:
                    rl_dict[receptor].append(idx)
        self.rl_dict = rl_dict

        # each receptor learns beta that is a function of its ligands
        receptor_beta_dicts = {
            rec: get_receptor_beta_model(rec_ligs) for rec, rec_ligs in rl_dict.items()}

        return receptor_beta_dicts

    def predict_y_from_tfs(self, model, betas, batch_labels, inputs_x, anchors):
        assert inputs_x.shape[1] == len(self.regulators) == model.dim-1
        assert betas.shape[1] == len(self.regulators)+1

        if anchors is None:
            anchors = np.stack(
                [self.beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
                axis=0
            )
        
        anchors = torch.from_numpy(anchors).float().to(device)
        inputs_x = inputs_x.to(device)
        
        y_pred = anchors[:, 0]*betas[:, 0]
         
        for w in range(model.dim-1):
            y_pred += anchors[:, w+1]*betas[:, w+1]*inputs_x[:, w]

        return y_pred

    def predict_y_from_lrs(self, dists, ligs, recs):
        # also need to create add coef aka beta_0
        ligs = ligs.to(device)
        recs = recs.to(device)
        batch_y = []
        for i, rec in enumerate(self.receptors):
            rl_model = self.receptor_beta_dicts[rec]
            rec_ligs = ligs[:, :, self.rl_dict[rec]] # batch, neighbors, ligand
            rbeta = rl_model(rec_ligs)
            batch_r = rbeta.flatten() * recs[:, i].flatten()
            batch_y.append(batch_r)

        batch_y = torch.stack(batch_y, dim=0)
        y_pred = torch.sum(batch_y, axis=0) # check these axes and such
        return y_pred # batch,


    def predict_y(self, model, betas, batch_labels, inputs_x,
                   dists, ligs, recs, anchors=None, lambd=0.2):
        y_tf = self.predict_y_from_tfs(model, betas, batch_labels, inputs_x, anchors)
        if not self.ligand_affected:
            return y_tf
        
        y_lr = self.predict_y_from_lrs(dists, ligs, recs)
        return y_tf*(1-lambd) + y_lr*(lambd)

    @torch.no_grad()
    def get_betas(self, xy=None, spatial_maps=None, labels=None, spatial_dim=None, beta_dists=None, layer=None):
        
        dataloader = self._build_dataloaders_from_adata(
            self.adata, self.target_gene, self.regulators, self.ligands, self.receptors, batch_size=1024, 
                mode='infer', rotate_maps=False, annot=self.annot, layer=self.layer, spatial_dim=self.spatial_dim)

        tf_beta_list = []
        rec_beta_list = []
        
        for tf_load, lr_load in dataloader:
            batch_spatial, batch_x, batch_y, batch_labels = tf_load 
            batch_dists, batch_ligs, batch_recs = lr_load

            # get TF betas 
            tf_betas = self.model(batch_spatial.to(device), batch_labels.to(device))
            tf_beta_list.extend(tf_betas.cpu().numpy())

            # get receptor betas from ligands
            ligs = batch_ligs.to(device)
            recs = batch_recs.to(device)
            batch_y = []

            batch_rbetas = []
            for i, rec in enumerate(self.receptors):
                rl_model = self.receptor_beta_dicts[rec]
                rec_ligs = ligs[:, :, self.rl_dict[rec]] # batch, neighbors, ligand
                rbeta = rl_model(rec_ligs)
                batch_rbetas.extend(rbeta)
            
            rec_beta_list.append(batch_rbetas)
        
        return torch.tensor(tf_beta_list), torch.tensor(rec_beta_list)


    def _training_loop(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        for tf_load, lr_load in dataloader:
            batch_spatial, batch_x, batch_y, batch_labels = tf_load 
            batch_dists, batch_ligs, batch_recs = lr_load
            optimizer.zero_grad()
            betas = model(batch_spatial.to(device), batch_labels.to(device))
            # if self.cluster_grn:
            #     betas = self._mask_betas(betas, batch_labels)
            anchors = np.stack(
                [self.beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
                axis=0
            )

            y_pred = self.predict_y(model, betas, batch_labels, batch_x, 
                                    batch_dists, batch_ligs, batch_recs, anchors)
            
            loss = criterion(y_pred.squeeze(), batch_y.to(device).squeeze())
            loss += 1e-3*((betas.mean(0) - torch.from_numpy(anchors).float().mean(0).to(device))**2).sum()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)

    @torch.no_grad()
    def _validation_loop(self, model, dataloader, criterion, cluster_grn = False):
        model.eval()
        total_loss = 0
        for tf_load, lr_load in dataloader:
            batch_spatial, batch_x, batch_y, batch_labels = tf_load 
            batch_dists, batch_ligs, batch_recs = lr_load

            betas = model(batch_spatial.to(device), batch_labels.to(device))
            
            # if cluster_grn:
            #     betas = self._mask_betas(betas, batch_labels)
            
            outputs = self.predict_y(model, betas, batch_labels, batch_x, 
                                    batch_dists, batch_ligs, batch_recs, anchors=None)
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def get_preds(self):
        dataloader = self._build_dataloaders_from_adata(
            self.adata, self.target_gene, self.regulators, self.ligands, self.receptors, batch_size=32, 
            mode='infer', rotate_maps=True, annot=self.annot, layer=self.layer, spatial_dim=self.spatial_dim)

        y_truths = []
        y_preds = []
        for tf_load, lr_load in dataloader:
            batch_spatial, batch_x, batch_y, batch_labels = tf_load 
            batch_dists, batch_ligs, batch_recs = lr_load

            betas = self.model(batch_spatial.to(device), batch_labels.to(device))
            outputs = self.predict_y(
                        self.model, betas, batch_labels, batch_x, 
                        batch_dists, batch_ligs, batch_recs, anchors=None)
            y_truths.extend(batch_y)
            y_preds.extend(outputs.flatten())
        
        return torch.tensor(y_truths), torch.tensor(y_preds)

    def _build_model(
        self,
        adata,
        annot,
        spatial_dim,
        mode,
        layer,
        max_epochs,
        batch_size,
        learning_rate,
        rotate_maps,
        cluster_grn=True,
        regularize=False,
        pbar=None
        ):

        train_dataloader, valid_dataloader = self._build_dataloaders_from_adata(
            adata, self.target_gene, self.regulators, self.ligands, self.receptors,
            mode=mode, rotate_maps=rotate_maps, 
            batch_size=batch_size, annot=annot, 
            layer=layer,
            spatial_dim=spatial_dim
        )


        model = NicheAttentionNetwork(
            n_regulators=len(self.regulators),
            in_channels=self.n_clusters,
            spatial_dim=spatial_dim,
        )


        model.to(device)

        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
    
        _prefix = f'[{self.target_gene} / {len(self.regulators)}]'

        if pbar is None:
            _manager = enlighten.get_manager()
            pbar = _manager.counter(
                total=max_epochs, 
                desc=f'{_prefix} <> MSE: ...', 
                unit='epochs'
            )
            pbar.refresh()
            
        for epoch in range(max_epochs):
            training_loss = self._training_loop(
                model, train_dataloader, criterion, optimizer)
            validation_loss = self._validation_loop(
                model, valid_dataloader, criterion)
            
            losses.append(validation_loss)

            if validation_loss < best_score:
                best_score = validation_loss
                best_model = copy.deepcopy(model)
                best_iter = epoch
            
            pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4g}'
            pbar.update()
            
        best_model.eval()
        
        return best_model, losses


    @staticmethod
    def _build_dataloaders_from_adata(adata, target_gene, regulators, ligands, receptors, batch_size=32, 
    mode='train', rotate_maps=True, annot='rctd_cluster', layer='imputed_count', spatial_dim=64, test_size=0.2):

        assert mode in ['train', 'train_test', 'infer']
        set_seed(42)

        xy = adata.obsm['spatial']
        labels = np.array(adata.obs[annot])
    
        g = torch.Generator()
        g.manual_seed(42)
        
        params = {
            'batch_size': batch_size,
            'worker_init_fn': seed_worker,
            'generator': g,
            'pin_memory': False,
            'num_workers': 0,
            'drop_last': True,
        }
        
        dataset = LigRecDataset(
            adata.copy(), 
            target_gene=target_gene, 
            regulators=regulators, 
            ligands=ligands,
            receptors=receptors,
            annot=annot, 
            layer=layer,
            spatial_dim=spatial_dim,
            rotate_maps=rotate_maps
        )

        if mode == 'infer':
            dataloader = DataLoader(dataset, shuffle=False, **params)
            return dataloader

        if mode == 'train':
            train_dataloader = DataLoader(dataset, shuffle=True, **params)
            valid_dataloader = DataLoader(dataset, shuffle=False, **params)
            return train_dataloader, valid_dataloader
        
        if mode == 'train_test':
            split = int((1-test_size)*len(dataset))
            generator = torch.Generator().manual_seed(42)
            train_dataset, valid_dataset = random_split(
                dataset, [split, len(dataset)-split], generator=generator)
            train_dataloader = DataLoader(train_dataset, shuffle=True, **params)
            valid_dataloader = DataLoader(valid_dataset, shuffle=False, **params)

            return train_dataloader, valid_dataloader