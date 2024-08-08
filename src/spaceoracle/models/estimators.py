import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pysal.model.spreg import OLS
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Normalize
from .spatial_map import xyc2spatial

from ..tools.utils import set_seed


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
    
    
class LeastSquaredEstimator(Estimator):
    
    def fit(self, X, y):
        ols_model = OLS(y=y, x=X)
        self.betas = ols_model.betas
    
    def get_betas(self):
        return self.betas
    

def _build_dataloaders(
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
        
        return total_loss / len(dataloader)
    
    
    def _estimate_baseline(self, dataloader, beta_init):
        total_linear_err = 0
        for _, batch_x, batch_y, _ in dataloader:
            _x = batch_x.cpu().numpy()
            _y = batch_y.cpu().numpy()
            
            ols_pred = beta_init[0]
            
            for w in range(len(beta_init)-1):
                ols_pred += _x[:, w]*beta_init[w+1]
                
            ols_err = np.mean((_y - ols_pred)**2)
            
            total_linear_err += ols_err
            
        return total_linear_err / len(dataloader)
        
    
            
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
            
                if np.mean(losses) < best_score:
                    best_model = copy.deepcopy(model)
                    best_iter = epoch
            
        best_model.eval()
        
        return best_model, losses
        
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
        
        ##TODO: make this more efficient
        y_pred = x[:, 0]*self.betas[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1]*inputs_x[:, w]*self.betas[w+1]

        return y_pred, x
