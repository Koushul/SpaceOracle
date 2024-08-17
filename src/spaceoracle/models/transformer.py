import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm 

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Normalize
from pysal.model.spreg import OLS

from .estimators import Estimator, LeastSquaredEstimator
from .spatial_map import xy2distance

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class TransformerEstimator(Estimator):

    def _build_dataloaders(
            self,
            X, y, xy,
            cluster_labels,
            context_length,
            mode='train',
            batch_size=24,
            test_size=0.2
            ):
        
        norm = Normalize(0, 1)

        distances = xy2distance(xy, context_length) # (cell, context_length)
        distances = torch.from_numpy(distances / np.max(distances) ).float()

        if mode == 'infer':
            dataset = TensorDataset(
                torch.from_numpy(X).float(),
                distances,
                torch.from_numpy(cluster_labels).long()
            )   
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if mode == 'train_test':
        
            dataset = TensorDataset(
                torch.from_numpy(X).float(),
                distances,
                torch.from_numpy(cluster_labels).long(),
                torch.from_numpy(y).float()
            )   
            
            split = int((1-test_size)*len(dataset))
            train_dataset, valid_dataset = random_split(dataset, [split, len(dataset)-split])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False)

            return train_dataloader, valid_dataloader
        
        
        if mode == 'train':
            dataset = TensorDataset(
                torch.from_numpy(X).float(),
                distances,
                torch.from_numpy(cluster_labels).long(),
                torch.from_numpy(y).float()
            )

            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            return train_dataloader, valid_dataloader



    def _build_model(
        self,
        X, y, xy,
        cluster_labels,
        beta_init, 
        context_length, 
        max_epochs,
        learning_rate,
        n_head=4, n_layer=4, dropout=0.1,
        mode='train'
        ):

        n_clusters = len(np.unique(cluster_labels))
        n_embd = 512

        model = Transformer(beta_init, n_embd, n_layer, n_head, n_clusters, dropout, context_length)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.to(device)

        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf

        train_dataloader, valid_dataloader = self._build_dataloaders(X, y, xy, cluster_labels, context_length, mode=mode)

        model.eval()

        with tqdm(range(max_epochs)) as pbar:
            for epoch in pbar:
                model.train()
                total_loss = 0
                for batch_x, batch_distances, batch_labels, batch_y in train_dataloader:
                    optimizer.zero_grad()

                    outputs, _ = model(*[x.to(device) for x in [batch_distances, batch_x, batch_labels]])
                    loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for batch_x, batch_distances, batch_labels, batch_y in train_dataloader:
                        outputs, _ = model(*[x.to(device) for x in [batch_distances, batch_x, batch_labels]])
                        loss = criterion(outputs.squeeze(), batch_y.to(device).squeeze())
                        total_loss += loss.item()

                    avg_validation_loss = total_loss / len(valid_dataloader)
                
                losses.append(avg_validation_loss)

                pbar.set_description(f'MSE: {np.mean(losses):.4f}')
            
                if np.mean(losses) < best_score:
                    best_model = copy.deepcopy(model)
            
        best_model.eval()
        
        return best_model, losses



    def fit(
        self,
        X, y, xy, 
        cluster_labels,
        init_betas='ols', 
        max_epochs=100, 
        learning_rate=0.001, 
        context_length=5,
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
            
        self.beta_init = beta_init.reshape(-1, )


        try:
            model, losses = self._build_model(
                X, y, xy,
                cluster_labels,
                beta_init, 
                context_length=context_length, 
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                mode=mode
            ) 
            
            self.model = model  
            self.losses = losses
        
        except KeyboardInterrupt:
            print('Training interrupted...')
            pass

    @torch.no_grad()
    def get_betas(self, X, xy, cluster_labels):
        infer_dataloader = self._build_dataloaders(
            X=X, y=None, xy=xy, cluster_labels=cluster_labels, context_length=self.model.context_length, mode='infer')
        beta_list = []
        y_pred = []
        for batch_x, batch_distances, batch_labels in infer_dataloader:
            outputs, betas = self.model(*[x.to(device) for x in [batch_distances, batch_x, batch_labels]])
            beta_list.extend(betas.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
        return np.array(beta_list), np.array(y_pred)



### Transformer Blocks ###

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


### Transformer Model ###

class Transformer(nn.Module):
    def __init__(self, betas, n_embd, n_layer, n_head, n_clusters, dropout, block_size):
        super().__init__()
        self.dim = betas.shape[0]
        self.context_length = block_size
        self.betas = torch.tensor(betas, dtype=torch.float32).to(device)

        # each token directly reads off the logits for the next token from a lookup table
        self.distance_token_table = nn.Linear(block_size, n_embd)
        self.celltype_embedding_table = nn.Embedding(n_clusters, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        
        self.lm_head = nn.Linear(n_embd, self.dim) # linear layer for prediction
    
    def forward(self, inputs_dis, inputs_x, inputs_labels):
        B, T = inputs_dis.shape

        tok_emb = self.distance_token_table(inputs_dis).squeeze()
        pos_emb = self.celltype_embedding_table(inputs_labels).squeeze()

        x = tok_emb + pos_emb # (B,T)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = self.lm_head(x).squeeze()

        y_pred = x[:, 0]*self.betas[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1] * inputs_x[:, w].long() * self.betas[w+1]

        return y_pred, x


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
    
    estimator = TransformerEstimator()
    print('Fitting...')
    estimator.fit(X, y, xy, c)
    print(estimator.get_betas().shape)
    
    # y_pred = estimator.predict(X, xy)
    # print(mean_squared_error(y, y_pred))