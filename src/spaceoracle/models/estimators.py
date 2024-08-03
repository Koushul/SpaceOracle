import numpy as np
from pysal.model.spreg import OLS
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Normalize

from .spatial_map import xy2spatial


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
    
    
    
class GeoCNNEstimator(Estimator):
    
    
    def _build_dataloaders(self, X, y, xy, spatial_dim, batch_size=32, test_size=0.2):
        norm = Normalize(0, 1)
        
        spatial_maps = xy2spatial(xy[:, 0], xy[:, 1], spatial_dim, spatial_dim)
        spatial_maps = spatial_maps/spatial_maps.mean()
        spatial_maps = norm(torch.from_numpy(spatial_maps)).unsqueeze(3)
        
        
        dataset = TensorDataset(
            spatial_maps.permute(0, 3, 1, 2).float(), 
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float()
        )   
        
        split = int((1-test_size)*len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [split, len(dataset)-split])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

        return train_dataloader, valid_dataloader
        
    def _build_cnn(self, X, y, xy, beta_init, in_channels, init, spatial_dim, max_epochs, learning_rate):
        model = GCNNWR(beta_init, in_channels=in_channels, init=init)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.to(device)
        # model = torch.compile(model)
        
        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
        
        
        train_dataloader, valid_dataloader = self._build_dataloaders(X, y, xy, spatial_dim)
        
        with tqdm(range(max_epochs)) as pbar:
            for epoch in pbar:
                model.train()
                total_loss = 0
                for batch_spatial, batch_x, batch_y in train_dataloader:
                    
                    optimizer.zero_grad()
            
                    outputs, _ = model(batch_spatial.to(device), batch_x.to(device))
                    loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                avg_training_loss = total_loss / len(train_dataloader)

                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for batch_spatial, batch_x, batch_y in valid_dataloader:
                        outputs, _ = model(batch_spatial.to(device), batch_x.to(device))
                        loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
                        total_loss += loss.item()
                    
                    avg_validation_loss = total_loss / len(valid_dataloader)
                
                losses.append(avg_validation_loss)
    
            if np.mean(losses) < best_score:
                best_model = copy.deepcopy(model)
                best_iter = epoch
            
            pbar.update()
            pbar.set_description(f'{criterion.__class__.__name__}: {np.mean(losses):.4f}')
            
        return best_model, losses
        
    def fit(self, X, y, xy, init_betas='ols', spatial_dim=64, in_channels=1, init=0.1):
        assert init_betas in ['ones', 'ols']
        assert X.shape[0] == y.shape[0] == xy.shape[0]
        
        
        if init_betas == 'ones':
            beta_init = torch.ones(X.shape[1]+1).reshape(-1, )
        
        elif init_betas == 'ols':
            ols = LeastSquaredEstimator()
            ols.fit(X, y)
            beta_init = ols.get_betas().reshape(-1, )
            
        print(beta_init)
        
        self._build_cnn(
            X, y,
            xy,
            beta_init, 
            in_channels=in_channels, 
            init=init, 
            spatial_dim=spatial_dim, 
            max_epochs=100,
            learning_rate=0.001
        )      
        
    def get_betas(self):
        return None



class GCNNWR(nn.Module):
    def __init__(self, betas, in_channels=1, init=0.1):
        super(GCNNWR, self).__init__()
        self.dim = betas.shape[0]
        self.betas = list(betas)
        
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
            
            nn.Linear(16, self.dim+1)
        )


    def forward(self, inputs_dis, inputs_x):
        x = self.conv_layers(inputs_dis)
        x = self.fc_layers(x)

        ##TODO: make this more efficient
        y_pred = x[:, 0]*self.betas[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1]*inputs_x[:, w]*self.betas[w+1]

        return y_pred, x
    
    
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1, )
    xy = np.random.rand(1000, 2)
    
    estimator = GeoCNNEstimator()
    print('Fitting...')
    estimator.fit(X, y, xy)
    print(estimator.get_betas().shape)
    
    # y_pred = estimator.predict(X, xy)
    # print(mean_squared_error(y, y_pred))