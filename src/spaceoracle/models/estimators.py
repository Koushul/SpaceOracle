import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pysal.model.spreg import OLS
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm 
import enlighten
from numba import jit

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Normalize
from .spatial_map import xyc2spatial
from .vit_blocks import ViT

from ..tools.utils import set_seed, seed_worker, deprecated
from ..tools.data import SpaceOracleDataset
from ..tools.network import GeneRegulatoryNetwork, DayThreeRegulatoryNetwork

set_seed(42)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

norm = Normalize(0, 1)


class Estimator(ABC):
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def get_betas(self):
        pass
    
class LeastSquaredEstimator(Estimator):
    
    def fit(self, X, y):
        ols_model = OLS(y=y, x=X)
        self.betas = ols_model.betas
        self.pvals = np.array(ols_model.t_stat)[:, 1]
    
    def get_betas(self):
        return self.betas

class ClusterLeastSquaredEstimator(LeastSquaredEstimator):
    
    def fit(self, X, y, clusters):
        self.beta_dict = {}
        self.pval_dict = {}
        self.clusters = clusters
        for cluster_label in np.unique(clusters):
            ols_model = OLS(y=y[clusters==cluster_label], x=X[clusters==cluster_label])
            self.beta_dict[cluster_label] = ols_model.betas
            self.pval_dict[cluster_label] = np.array(ols_model.t_stat)[:, 1]


    def get_betas(self, cluster_label):
        return self.beta_dict[self.betas]

class BetaModel(nn.Module):
    @deprecated('Please use ViT instead.')
    def __init__(self, betas, in_channels=1, init=0.1):
        set_seed(42)
        super(BetaModel, self).__init__()
        self.dim = betas.shape[0]
        self.betas = torch.tensor(betas.astype(np.float32)).to(device)
        
        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3, padding='same')),
            nn.PReLU(init=init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
            nn.PReLU(init=init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(64, 256, kernel_size=3, padding='same')),
            nn.PReLU(init=init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(init=init),
            
            nn.Linear(128, 64),
            nn.PReLU(init=init),
            
            nn.Linear(64, 16),
            nn.PReLU(init=init),
            nn.Dropout(0.2),
            nn.Linear(16, self.dim)
        )


    def forward(self, spatial_map, input_labels):
        spatial_features = self.conv_layers(spatial_map)
        betas = self.fc_layers(spatial_features)

        return betas


class VisionEstimator(Estimator):
    def __init__(self, adata, target_gene, regulators=None, n_clusters=None, layer='imputed_count'):
        assert target_gene in adata.var_names
        assert layer in adata.layers

        self.adata = adata
        self.target_gene = target_gene
        # self.grn = GeneRegulatoryNetwork()
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN

        if regulators == None and n_clusters == None:
            self.regulators = self.grn.get_cluster_regulators(self.adata, self.target_gene)
            self.n_clusters = len(self.adata.obs['rctd_cluster'].unique())
        else:
            self.regulators = regulators
            self.n_clusters = n_clusters
        
        self.layer = layer
        self.model = None
        self.losses = []
        
        # assert len(self.regulators) > 0, f'No regulators found for target gene {self.target_gene}.'

    def predict_y(self, model, betas, inputs_x):

        assert inputs_x.shape[1] == len(self.regulators) == model.dim-1
        assert betas.shape[1] == len(self.regulators)+1 == len(model.betas) == len(self.regulators)+1

        y_pred = betas[:, 0]*model.betas[0]
         
        for w in range(model.dim-1):
            y_pred += betas[:, w+1]*inputs_x[:, w]*model.betas[w+1]

        return y_pred
    
    def _mask_betas(self, betas, batch_labels):
        regulator_dict = self.grn.regulator_dict
        relevant_tfs = [regulator_dict[label.item()] for label in batch_labels]
        
        mask = torch.stack(relevant_tfs).to(device)
        return betas * mask

    def _training_loop(self, model, dataloader, criterion, optimizer, cluster_grn=False, regularize=False, lambd=1e-3, a=0.9):
        model.train()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            
            optimizer.zero_grad()
            betas = model(batch_spatial.to(device), batch_labels.to(device))

            if cluster_grn:
                betas = self._mask_betas(betas, batch_labels)
            
            outputs = self.predict_y(model, betas, inputs_x=batch_x.to(device))

            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            if regularize: ##TODO: make this work more consistently
                loss += lambd * ((a*torch.sum((betas)**2) + ((1-a)/2)*torch.sum(abs(betas)) ))
                # loss += scale * torch.sum((betas)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def _validation_loop(self, model, dataloader, criterion, cluster_grn = False):
        model.eval()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            betas = model(batch_spatial.to(device), batch_labels.to(device))
            
            if cluster_grn:
                betas = self._mask_betas(betas, batch_labels)
            
            outputs = self.predict_y(model, betas, inputs_x=batch_x.to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _estimate_baseline(self, dataloader, beta_init):
        total_linear_err = 0
        torch.manual_seed(42)
        for _, batch_x, batch_y, _ in dataloader:
            _x = batch_x.cpu().numpy()
            _y = batch_y.cpu().numpy()
            
            ols_pred = beta_init[0]

            
            for w in range(len(beta_init)-1):
                ols_pred += _x[:, w]*beta_init[w+1]
                
            ols_err = np.mean((_y - ols_pred)**2)
            
            total_linear_err += ols_err
            
        return total_linear_err / len(dataloader)
        
    @staticmethod
    def _build_dataloaders_from_adata(adata, target_gene, regulators, batch_size=32, 
    mode='train', rotate_maps=True, annot='rctd_cluster', layer='imputed_count', spatial_dim=64, test_size=0.2):

        assert mode in ['train', 'train_test']
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
        
        dataset = SpaceOracleDataset(
            adata.copy(), 
            target_gene=target_gene, 
            regulators=regulators, 
            annot=annot, 
            layer=layer,
            spatial_dim=spatial_dim,
            rotate_maps=rotate_maps
        )
        

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
        
        
    @torch.no_grad()
    def get_betas(self, xy, labels, spatial_dim=None, layer=None):
        """
        Get beta values for the given spatial coordinates and labels.

        This method processes the input spatial data and labels through the trained model
        to obtain beta values, which represent the importance of each regulator for the target gene.

        Parameters:
        -----------
        xy : numpy.ndarray
            Array of shape (n_samples, 2) containing spatial coordinates.
        labels : numpy.ndarray
            Array of shape (n_samples,) containing cell type or cluster labels.
        spatial_dim : int, optional
            Dimension of the spatial map. If None, uses the spatial dimension used to train the model.

        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, n_regulators) containing beta values for each sample and regulator.


        """

        spatial_dim = self.spatial_dim if spatial_dim is None else spatial_dim
        
        spatial_maps = norm(
            torch.from_numpy(
                xyc2spatial(xy[:, 0], xy[:, 1], labels, spatial_dim, spatial_dim)
            ).float()
        )

        dataset = TensorDataset(
            spatial_maps.float(), 
            torch.from_numpy(labels).long()
        )   

        g = torch.Generator()
        g.manual_seed(42)
        
        params = {
            'batch_size': 64,
            'worker_init_fn': seed_worker,
            'generator': g
        }
        
        infer_dataloader = DataLoader(dataset, shuffle=False, **params)

        if self.cluster_grn:
            beta_stack = []
            
            for batch_spatial, batch_labels in infer_dataloader:
                betas = self.model(batch_spatial.to(device), batch_labels.to(device))
                betas = self._mask_betas(betas, batch_labels)
                beta_stack.append(betas)
        
            beta_stack = torch.cat(beta_stack)
            return beta_stack.cpu().numpy()

        else:
            beta_list = []
            
            for batch_spatial, batch_labels in infer_dataloader:
                betas = self.model(batch_spatial.to(device), batch_labels.to(device))
                beta_list.extend(betas.cpu().numpy())
            
            return np.array(beta_list)

class GeoCNNEstimatorV2(VisionEstimator):
    def _build_cnn(
        self, 
        adata,
        annot,
        spatial_dim,
        mode, 
        max_epochs,
        batch_size, 
        learning_rate,
        rotate_maps
        ):


        train_dataloader, valid_dataloader = self._build_dataloaders_from_adata(
                adata, self.target_gene, self.regulators, 
                mode=mode, rotate_maps=rotate_maps, batch_size=batch_size,
                annot=annot, spatial_dim=spatial_dim)
           
        model = BetaModel(self.beta_init, in_channels=self.n_clusters)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        model.to(device)
        # model = torch.compile(model)
        
        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
    
        baseline_loss = self._estimate_baseline(valid_dataloader, self.beta_init)
            
        with tqdm(range(max_epochs)) as pbar:
            for epoch in pbar:
                training_loss = self._training_loop(model, train_dataloader, criterion, optimizer)
                validation_loss = self._validation_loop(model, valid_dataloader, criterion)
                
                losses.append(validation_loss)

                pbar.set_description(f'[{device.type}] MSE: {np.mean(losses):.4f} | Baseline: {baseline_loss:.4f}')
            
                if validation_loss < best_score:
                    best_score = validation_loss
                    best_model = copy.deepcopy(model)
                    best_iter = epoch
            
        best_model.eval()
        
        print(f'Best model at {best_iter}/{max_epochs}')
        
        return best_model, losses
        
    def fit(
        self,
        annot,
        init_betas='ols', 
        max_epochs=100, 
        learning_rate=0.001, 
        spatial_dim=64,
        batch_size=32, 
        mode='train',
        rotate_maps=True
        ):
        
        
        assert init_betas in ['ones', 'ols']
        
        self.spatial_dim = spatial_dim  

        adata = self.adata.copy()

        if init_betas == 'ones':
            beta_init = torch.ones(len(self.regulators)+1)
        
        elif init_betas == 'ols':
            X = adata.to_df()[self.regulators].values
            y = adata.to_df()[[self.target_gene]].values
            ols = LeastSquaredEstimator()
            ols.fit(X, y)
            beta_init = ols.get_betas()
            
        self.beta_init = np.array(beta_init).reshape(-1, )
        
        try:
            model, losses = self._build_cnn(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rotate_maps=rotate_maps
            ) 
            
            self.model = model  
            self.losses = losses
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass
        
        

class SpatialInsights(VisionEstimator):
    def _build_model(
        self,
        adata,
        annot,
        spatial_dim,
        mode,
        max_epochs,
        batch_size,
        learning_rate,
        rotate_maps,
        regularize,
        cluster_grn,
        n_patches=2, n_blocks=4, hidden_d=14, n_heads=2,
        pbar=None
        ):

        train_dataloader, valid_dataloader = self._build_dataloaders_from_adata(
                adata, self.target_gene, self.regulators, 
                mode=mode, rotate_maps=rotate_maps, batch_size=batch_size,
                annot=annot, layer=self.layer, spatial_dim=spatial_dim)
        
        # self.train_dataloader = train_dataloader
        # self.valid_dataloader = valid_dataloader

        model = ViT(
            self.beta_init, 
            in_channels=self.n_clusters, 
            spatial_dim=spatial_dim, 
            n_patches=n_patches, 
            n_blocks=n_blocks, 
            hidden_d=hidden_d, 
            n_heads=n_heads
        )
        
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        model.to(device)
        # model = torch.compile(model)
        
        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
    
        baseline_loss = self._estimate_baseline(valid_dataloader, self.beta_init)
        _prefix = f'[{self.target_gene} / {len(self.regulators)}]'

        if pbar is None:
            _manager = enlighten.get_manager()
            pbar = _manager.counter(
                total=max_epochs, 
                desc=f'{_prefix} <> MSE: ... | Baseline: {baseline_loss:.4f}', 
                unit='epochs'
            )
            pbar.refresh()
            
        for epoch in range(max_epochs):
            training_loss = self._training_loop(model, train_dataloader, criterion, optimizer, cluster_grn=cluster_grn, regularize=regularize)
            validation_loss = self._validation_loop(model, valid_dataloader, criterion, cluster_grn=cluster_grn)
            
            losses.append(validation_loss)

            if validation_loss < best_score:
                best_score = validation_loss
                best_model = copy.deepcopy(model)
                best_iter = epoch
            
            # pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4f} (*{best_iter}*) | Baseline: {baseline_loss:.4f}'
            pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4f} (*{best_iter}*) | alpha: {model.alpha.item():.4f}'
            # pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4f} (*{best_iter}*) | alpha: {model.alpha:.4f}'


            pbar.update()
            
        best_model.eval()
        
        # print(f'Best model at {best_iter}/{max_epochs}')
        
        return best_model, losses

    
    def fit(
        self,
        annot,
        init_betas='ols', 
        max_epochs=100, 
        learning_rate=2e-4, 
        spatial_dim=64,
        batch_size=32, 
        mode='train',
        regularize=False,
        cluster_grn=False,
        rotate_maps=True,
        n_patches=2, n_blocks=2, hidden_d=16, n_heads=2,
        pbar=None
        ):
        
        assert annot in self.adata.obs.columns
        assert init_betas in ['ones', 'ols', 'co']

        self.spatial_dim = spatial_dim  
        self.regularize = regularize
        self.cluster_grn = cluster_grn
        self.rotate_maps = rotate_maps
        self.init_betas = init_betas
        self.annot = annot

        adata = self.adata.copy()

        if init_betas == 'ones':
            beta_init = torch.ones(len(self.regulators)+1)
        
        elif init_betas == 'ols':
            X = adata.to_df(layer=self.layer)[self.regulators].values
            y = adata.to_df(layer=self.layer)[[self.target_gene]].values
            ols = LeastSquaredEstimator()
            ols.fit(X, y)
            beta_init = ols.get_betas()

        elif init_betas == 'co':
            co_coefs = self.grn.get_regulators_with_pvalues(adata, self.target_gene).groupby('source').mean()
            co_coefs = co_coefs.loc[self.regulators]
            beta_init = np.array(co_coefs.values).reshape(-1, )
            beta_init = np.concatenate([beta_init, [1]], axis=0) 
            
        self.beta_init = np.array(beta_init).reshape(-1, )

        assert len(self.beta_init) == len(self.regulators)+1
        
        try:
            model, losses = self._build_model(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rotate_maps=rotate_maps,
                regularize=regularize,
                cluster_grn=cluster_grn,
                n_patches=n_patches, 
                n_blocks=n_blocks, 
                hidden_d=hidden_d, 
                n_heads=n_heads,
                pbar=pbar
            )
            
            self.model = model  
            self.losses = losses
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass


    def export(self):
        self.model.eval()
        # self.model.cpu()
        return self.model, self.regulators, self.target_gene


## backward compatibility
ViTEstimatorV2 = SpatialInsights
