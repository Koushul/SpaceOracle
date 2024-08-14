import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pysal.model.spreg import OLS
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm 
import wandb

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Normalize
from .spatial_map import xyc2spatial
from .vit_blocks import ViT

from ..tools.utils import set_seed, seed_worker, deprecated
from ..tools.data import SpaceOracleDataset
from ..tools.network import GeneRegulatoryNetwork

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
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def get_betas(self):
        pass

class VisionEstimator(Estimator):
    
    @torch.no_grad()
    def get_betas(self, X, xy, labels, spatial_dim):
        infer_dataloader = self._build_dataloaders(
            X=X, y=None, xy=xy, labels=labels, spatial_dim=spatial_dim, mode='infer')
        
        beta_list = []
        y_pred = []
        
        for batch_spatial, batch_x, batch_labels in infer_dataloader:
            outputs, betas = self.model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            beta_list.extend(betas.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
        return np.array(beta_list), np.array(y_pred)

    
    def _build_dataloaders(
        self,
        X, y, xy,
        labels, 
        spatial_dim, 
        mode='train', 
        batch_size=32, 
        test_size=0.2
        ):
        
        assert mode in ['train', 'infer', 'train_test']
        
        spatial_maps = norm(
            torch.from_numpy(
                xyc2spatial(xy[:, 0], xy[:, 1], labels, spatial_dim, spatial_dim)
            ).float()
        )
        
        if mode == 'infer':
            dataset = TensorDataset(
                spatial_maps.float(), 
                torch.from_numpy(X).float(),
                torch.from_numpy(labels).long()
            )   
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # otherwise
        
        dataset = TensorDataset(
            spatial_maps.float(), 
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(labels).long()
        )  
        
        if mode == 'train':
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            return train_dataloader, valid_dataloader
        
        if mode == 'train_test':
            split = int((1-test_size)*len(dataset))
            generator = torch.Generator().manual_seed(42)
            train_dataset, valid_dataset = random_split(
                dataset, [split, len(dataset)-split], generator=generator)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False)

            return train_dataloader, valid_dataloader
    
    
    def _estimate_baseline(self, dataloader, beta_init):
        total_linear_err = 0
        for _, batch_x, batch_y, _ in dataloader:
            _x = batch_x.cpu().numpy()
            _y = batch_y.cpu().numpy()
            
            ols_pred = beta_init[0].item()
            
            for w in range(len(beta_init)-1):
                # print(ols_pred.shape, _x.shape, _x[:, w].shape , beta_init[w+1].shape )

                ols_pred += _x[:, w] * beta_init[w+1].item()
                
            ols_err = np.mean((_y - ols_pred)**2)
            
            total_linear_err += ols_err
            
        return total_linear_err / len(dataloader)
    
    def _training_loop(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)
    
    
    @torch.no_grad()
    def _validation_loop(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            outputs, _ = model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    
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
    
@deprecated('Please use the SpatialDataset class instead.')
def _build_dataloaders(
    X, y, xy,
    labels, 
    spatial_dim, 
    mode='train', 
    batch_size=32, 
    test_size=0.2
    ):
    
    assert mode in ['train', 'infer', 'train_test']
    set_seed(42)
    

    spatial_maps = torch.from_numpy(
            xyc2spatial(xy[:, 0], xy[:, 1], labels, spatial_dim, spatial_dim)
        ).float()
    
    
    g = torch.Generator()
    g.manual_seed(42)
    
    params = {
        'batch_size': batch_size,
        'worker_init_fn': seed_worker,
        'generator': g
    }
    
    
    if mode == 'infer':
        dataset = TensorDataset(
            spatial_maps.float(), 
            torch.from_numpy(X).float(),
            torch.from_numpy(labels).long()
        )   
        
        return DataLoader(dataset, shuffle=False, **params)
    
    # otherwise
    
    dataset = TensorDataset(
        spatial_maps.float(), 
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(labels).long()
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


class GeoCNNEstimator(Estimator):
    def _training_loop(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)
    
    
    @torch.no_grad()
    def _validation_loop(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            outputs, _ = model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            total_loss += loss.item()
        
        model.to(device)

        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
        
        train_dataloader, valid_dataloader = self._build_dataloaders(
            X, y, xy, labels, spatial_dim, mode=mode)
    
        baseline_loss = self._estimate_baseline(valid_dataloader, beta_init)
            
        with tqdm(range(max_epochs)) as pbar:
            for epoch in pbar:
                training_loss = self._training_loop(model, train_dataloader, criterion, optimizer)
                validation_loss = self._validation_loop(model, valid_dataloader, criterion)
                
                losses.append(validation_loss)

                pbar.set_description(f'[{device.type}] MSE: {np.mean(losses):.4f} | Baseline: {baseline_loss:.4f}')
            
                if np.mean(losses) < best_score:
                    best_model = copy.deepcopy(model)
                    best_iter = epoch
            
        best_model.eval()
        
        return best_model, losses
    
    def fit(self, 
        X, y, xy, 
        labels,
        init_betas='ols', 
        max_epochs=100, 
        learning_rate=0.001, 
        spatial_dim=64, 
        init=0.1,
        mode='train'
        ):

        in_channels = len(np.unique(labels))
        
        assert init_betas in ['ones', 'ols', 'random']
        assert X.shape[0] == y.shape[0] == xy.shape[0]
        
        
        if init_betas == 'ones':
            beta_init = torch.ones(X.shape[1]+1)
        
        elif init_betas == 'ols':
            ols = LeastSquaredEstimator()
            ols.fit(X, y)
            beta_init = ols.get_betas()
            
        elif init_betas == 'random':
            beta_init = torch.randn(X.shape[1]+1)
            
        self.beta_init = np.array(beta_init).reshape(-1, )
        
        
        try:
            model, losses = self._build_model(
                X, y, xy,
                labels,
                self.beta_init, 
                in_channels=in_channels, 
                spatial_dim=spatial_dim, 
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                mode=mode,
            ) 
            
            self.model = model  
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass
        
        self.losses = losses
    
    

    
class GeoCNNEstimator(VisionEstimator):
        
    def _build_cnn(
        self, 
        X, y, xy, 
        labels,
        beta_init, 
        in_channels, 
        init, 
        spatial_dim,
        mode, 
        max_epochs, 
        learning_rate
        ):
        
           
        model = GCNNWR(beta_init, in_channels=in_channels, init=init)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        model.to(device)
        # model = torch.compile(model)
        
        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
        
        train_dataloader, valid_dataloader = _build_dataloaders(
                    X, y, xy, labels, spatial_dim, mode=mode)
    
        baseline_loss = self._estimate_baseline(valid_dataloader, beta_init)
            
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
        
    @deprecated('Please use GeoCNNEstimatorV2 instead.')
    def fit(
        self, 
        X, y, xy, 
        labels,
        init_betas='ols', 
        max_epochs=100, 
        learning_rate=0.001, 
        spatial_dim=64, 
        in_channels=1, 
        init=0.1,
        mode='train'
        ):
        
        assert init_betas in ['ones', 'ols', 'random']
        assert X.shape[0] == y.shape[0] == xy.shape[0]
        
        
        if init_betas == 'ones':
            beta_init = torch.ones(X.shape[1]+1)
        
        elif init_betas == 'ols':
            ols = LeastSquaredEstimator()
            ols.fit(X, y)
            beta_init = ols.get_betas()
            
        elif init_betas == 'random':
            beta_init = torch.randn(X.shape[1]+1)
            
        self.beta_init = np.array(beta_init).reshape(-1, )
        
        
        try:
            model, losses = self._build_cnn(
                X, y,
                xy,
                labels,
                self.beta_init, 
                in_channels=in_channels, 
                init=init, 
                spatial_dim=spatial_dim, 
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                mode=mode,
            ) 
            
            self.model = model  
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass
        
        self.losses = losses
        
        
        
    @torch.no_grad()
    def get_betas(self, X, xy, labels, spatial_dim=64):
        infer_dataloader = _build_dataloaders(
            X=X, y=None, xy=xy, labels=labels, spatial_dim=spatial_dim, mode='infer')
        
        beta_list = []
        y_pred = []
        
        for batch_spatial, batch_x, batch_labels in infer_dataloader:
            outputs, betas = self.model(
                batch_spatial.to(device), batch_x.to(device), batch_labels.to(device))
            beta_list.extend(betas.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
        return np.array(beta_list), np.array(y_pred)



class GCNNWR(nn.Module):
    def __init__(self, betas, use_labels=True, in_channels=1, init=0.1):
        set_seed(42)
        super(GCNNWR, self).__init__()
        self.dim = betas.shape[0]
        self.betas = list(betas)
        self.use_labels = use_labels
        
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


    def forward(self, inputs_dis, inputs_x, inputs_labels):
        x = self.conv_layers(inputs_dis)
        x = self.fc_layers(x)
        
        y_pred = x[:, 0]*self.betas[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1]*inputs_x[:, w]*self.betas[w+1]

        return y_pred, x


    
class BetaModel(nn.Module):
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
    def __init__(self, adata, target_gene):
        assert target_gene in adata.var_names
        self.adata = adata
        self.target_gene = target_gene
        self.grn = GeneRegulatoryNetwork()
        self.regulators = self.grn.get_regulators(self.adata, self.target_gene)
        self.n_clusters = len(self.adata.obs['rctd_cluster'].unique())

    def predict_y(self, model, betas, inputs_x):
        y_pred = betas[:, 0]*model.betas[0]
         
        for w in range(model.dim-1):
            y_pred += betas[:, w+1]*inputs_x[:, w]*model.betas[w+1]

        return y_pred

    def _training_loop(self, model, dataloader, criterion, optimizer, regularize=False, lambd=0.01, a=0.9):
        model.train()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            optimizer.zero_grad()
            betas = model(batch_spatial.to(device), batch_labels.to(device))
            outputs = self.predict_y(model, betas, inputs_x=batch_x.to(device))

            loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
            if regularize:
                loss += lambd * ((a*torch.sum((betas)**2) + ((1-a)/2)*torch.sum(abs(betas)) ))
                # loss += scale * torch.sum((betas)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                    
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def _validation_loop(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0
        for batch_spatial, batch_x, batch_y, batch_labels in dataloader:
            betas = model(batch_spatial.to(device), batch_labels.to(device))
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
    mode='train', rotate_maps=True, annot='rctd_cluster', spatial_dim=64, test_size=0.2):

        assert mode in ['train', 'train_test']
        set_seed(42)

        xy = adata.obsm['spatial']
        labels = np.array(adata.obs[annot])
    
        g = torch.Generator()
        g.manual_seed(42)
        
        params = {
            'batch_size': batch_size,
            'worker_init_fn': seed_worker,
            'generator': g
        }
        
        dataset = SpaceOracleDataset(
            adata.copy(), 
            target_gene=target_gene, 
            regulators=regulators, 
            annot=annot, 
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
    def get_betas(self, xy, labels, spatial_dim=None):

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
        rotate_maps,
        regularize, lambd, a,
        use_wandb
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
                training_loss = self._training_loop(model, train_dataloader, criterion, optimizer, regularize=regularize, lambd=lambd, a=a)
                validation_loss = self._validation_loop(model, valid_dataloader, criterion)
                
                losses.append(validation_loss)
                if use_wandb:
                    wandb.log({'training_loss': training_loss, 'validation_loss': validation_loss})

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
        rotate_maps=True,
        regularize=False,
        lambd=0.01, a=0.5,
        use_wandb=False
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
                rotate_maps=rotate_maps,
                regularize=regularize, lambd=lambd, a=a,
                use_wandb=use_wandb
            ) 
            
            self.model = model  
            self.losses = losses
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass
        
        

class ViTEstimatorV2(VisionEstimator):
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
        n_patches, n_blocks, hidden_d, n_heads,
        lambd, a, use_wandb
        ):

        train_dataloader, valid_dataloader = self._build_dataloaders_from_adata(
                adata, self.target_gene, self.regulators, 
                mode=mode, rotate_maps=rotate_maps, batch_size=batch_size, annot=annot, spatial_dim=spatial_dim)
           
        model = ViT(self.beta_init, in_channels=self.n_clusters, spatial_dim=spatial_dim, 
                n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads)
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
                training_loss = self._training_loop(model, train_dataloader, criterion, optimizer, regularize=regularize, lambd=lambd, a=a)
                validation_loss = self._validation_loop(model, valid_dataloader, criterion)

                if use_wandb:
                    wandb.log({'training_loss': training_loss, 'validation_loss': validation_loss})
                
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
        regularize=True,
        rotate_maps=True,
        n_patches=16, n_blocks=2, hidden_d=8, n_heads=2,
        lambd=0.01, a=0.5,
        use_wandb=False
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
                n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads,
                lambd=lambd, a=a, 
                use_wandb=use_wandb
                )
            
            self.model = model  
            self.losses = losses
            
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass





if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('../src')

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1, )
    xy = np.random.rand(1000, 2)
    c = np.random.randint(0, 13, size=(1000, 1))


    estimator = ViTEstimator()
    print('Fitting...')
    estimator.fit(X, y, xy, c)
    print(estimator.get_betas().shape)



    # y_pred = estimator.predict(X, xy)
    # print(mean_squared_error(y, y_pred))