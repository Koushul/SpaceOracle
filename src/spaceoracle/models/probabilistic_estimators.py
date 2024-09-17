from spaceoracle.models.base_estimators import BayesianRegression, device
from spaceoracle.models.estimators import VisionEstimator
from .pixel_attention import NicheAttentionNetwork

import torch
import torch.nn as nn
import numpy as np
import copy
import enlighten
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO


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
        cluster_grn,
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
                model, train_dataloader, criterion, optimizer, 
                cluster_grn=cluster_grn)
            validation_loss = self._validation_loop(
                model, valid_dataloader, criterion, 
                cluster_grn=cluster_grn)
            
            losses.append(validation_loss)

            if validation_loss < best_score:
                best_score = validation_loss
                best_model = copy.deepcopy(model)
                best_iter = epoch
            
            pbar.desc = f'{_prefix} <> MSE: {np.mean(losses):.4g}'
            pbar.update()
            
        best_model.eval()
        
        return best_model, losses
    
    def predict_y(self, model, betas, batch_labels, inputs_x):

        assert inputs_x.shape[1] == len(self.regulators) == model.dim-1
        assert betas.shape[1] == len(self.regulators)+1

        anchors = np.stack(
            [self.beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
            axis=0
        )

        anchors = torch.from_numpy(anchors).float().to(device)

        y_pred = anchors[:, 0]*betas[:, 0]
         
        for w in range(model.dim-1):
            y_pred += anchors[:, w+1]*betas[:, w+1]*inputs_x[:, w]

        return y_pred


    def fit(
        self,
        annot,
        max_epochs=10, 
        learning_rate=2e-4, 
        spatial_dim=64,
        batch_size=32, 
        num_samples=1000,
        mode='train_test',
        rotate_maps=True,
        cluster_grn=True,
        pbar=None
        ):
        
        assert annot in self.adata.obs.columns

        self.spatial_dim = spatial_dim  
        self.rotate_maps = rotate_maps
        self.annot = annot
        self.cluster_grn = cluster_grn

        adata = self.adata
            
        X = torch.from_numpy(adata.to_df(layer=self.layer)[self.regulators].values).float()
        y = torch.from_numpy(adata.to_df(layer=self.layer)[self.target_gene].values).float()
        cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long()
        
        self.beta_model = BayesianRegression(
            n_regulators=len(self.regulators), device=torch.device('cpu'))

        self.beta_model.fit(
            X, y, cluster_labels, 
            max_epochs=3000, learning_rate=3e-3, 
            num_samples=num_samples
        )


        self.beta_dists = {}
        for cluster in range(self.n_clusters):
            self.beta_dists[cluster] = self.beta_model.get_betas(
                X[cluster_labels==cluster].to(self.beta_model.device), 
                cluster=cluster, 
                num_samples=1000
            )


        del X, y, cluster_labels

        try:
            model, losses = self._build_model(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                layer=self.layer,
                cluster_grn=cluster_grn,
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
            pass


    def export(self):
        self.model.eval()
        return self.model, self.regulators, self.target_gene
