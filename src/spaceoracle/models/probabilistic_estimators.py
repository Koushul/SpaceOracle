from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import copy
import enlighten
import pyro
import torch
import pyro
import torch.nn as nn
import copy
import os
import pickle
from tqdm import tqdm
from sklearn.linear_model import BayesianRidge
from spaceoracle.models.spatial_map import xyc2spatial_fast
from spaceoracle.models.estimators import VisionEstimator
from spaceoracle.models.vit_blocks import ViT
from spaceoracle.tools.data import LigRecDataset
from .pixel_attention import NicheAttentionNetwork
from .bayesian_linear import BayesianRegression
from ..tools.utils import gaussian_kernel_2d, set_seed, seed_worker

set_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

available_cores = os.cpu_count()

pyro.clear_param_store()


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
        beta_dists_file = f"/tmp/{self.target_gene}_beta_dists.pkl"
        cache_exists = os.path.exists(beta_dists_file)
            
        X = torch.from_numpy(adata.to_df(layer=self.layer)[self.regulators].values).float()
        y = torch.from_numpy(adata.to_df(layer=self.layer)[self.target_gene].values).float()
        cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long()
        
        if cache_exists and cache:
            with open(beta_dists_file, 'rb') as f:
                self.beta_dists = pickle.load(f)

        else:
            self.beta_model = BayesianRegression(
                n_regulators=len(self.regulators), 
                device=torch.device('cpu') ## use cpu for better parallelization
            )

            _max_epochs = 1 if self.test_mode else 1000
            ns = 1 if self.test_mode else num_samples

            self.beta_model.fit(
                X, y, cluster_labels, 
                max_epochs=_max_epochs, learning_rate=3e-3, 
                num_samples=ns,
                parallel=parallel
            )

            self.beta_dists = {}
            for cluster in range(self.n_clusters):
                self.beta_dists[cluster] = self.beta_model.get_betas(
                    X[cluster_labels==cluster].to(self.beta_model.device), 
                    cluster=cluster, 
                    num_samples=ns
                )
        
            with open(beta_dists_file, 'wb') as f:
                pickle.dump(self.beta_dists, f)

        

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
            _max_epochs = 1 if self.test_mode else max_epochs

            model, losses = self._build_model(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                layer=self.layer,
                max_epochs=_max_epochs,
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



class ProbabilisticPixelModulators(ProbabilisticPixelAttention):
    

    @staticmethod
    def received_ligands(xy, lig_df, radius=200):
        # gex_df = adata.to_df(layer=layer)
        ligands = lig_df.columns
        gauss_weights = [
            gaussian_kernel_2d(
                xy[i], 
                xy, 
                radius=radius) for i in range(len(lig_df)
            )
        ]
        u_ligands = np.unique(ligands)
        wL = lambda x: np.array([(gauss_weights[i] * lig_df[x]).mean(0) for i in range(len(lig_df))])
        weighted_ligands = [wL(x) for x in tqdm(u_ligands)]

        return pd.DataFrame(
            weighted_ligands, 
            index=u_ligands, 
            columns=lig_df.index
        ).T




    def _build_dataloaders_from_adata(self,
        adata, target_gene, 
        regulators, ligands, receptors,
        batch_size=32, radius=200,
        mode='train', rotate_maps=True, 
        annot='rctd_cluster', layer='imputed_count', 
        spatial_dim=64, test_size=0.2
    ):

        assert mode in ['train', 'train_test']
        set_seed(42)

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
            adata, 
            target_gene=target_gene, 
            regulators=regulators, 
            ligands=ligands,
            receptors=receptors,
            radius=radius,
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
            adata, self.target_gene, self.regulators, self.ligands, self.receptors,
            mode=mode, rotate_maps=rotate_maps, radius=self.radius,
            batch_size=batch_size, annot=annot, 
            layer=layer,
            spatial_dim=spatial_dim
        )


        model = NicheAttentionNetwork(
            n_regulators=len(self.regulators)+len(self.ligands),
            in_channels=self.n_clusters,
            spatial_dim=spatial_dim,
        )

        # beta_init = torch.ones(len(self.regulators)+len(self.ligands)+1)
        # model = ViT(
        #     beta_init, 
        #     in_channels=self.n_clusters, 
        #     spatial_dim=spatial_dim, 
        #     n_patches=8, 
        #     n_blocks=4, 
        #     hidden_d=16, 
        #     n_heads=2
        # )

        model.to(device)

        # print(model)

        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses = []
        best_model = copy.deepcopy(model)
        best_score = np.inf
        best_iter = 0
    
        # baseline_loss = self._estimate_baseline(valid_dataloader, self.beta_init)
        _prefix = f'[{self.target_gene} / {len(self.regulators)}]'

        print(f'{len(self.regulators)} regulators + {len(self.lr["pairs"])} ligand-receptor pairs')

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

        assert inputs_x.shape[1] == len(self.regulators)+len(self.ligands) == model.dim-1
        assert betas.shape[1] == len(self.regulators)+len(self.ligands)+1

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
        radius=200,
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
        self.radius = radius

        adata = self.adata
        beta_dists_file = f"/tmp/{self.target_gene}_beta_dists.pkl"
        cache_exists = os.path.exists(beta_dists_file)
            
        X = torch.from_numpy(adata.to_df(layer=self.layer)[self.regulators].values).float()
        y = torch.from_numpy(adata.to_df(layer=self.layer)[self.target_gene].values).float()
        cluster_labels = torch.from_numpy(np.array(adata.obs[self.annot])).long()
        
        # Generate ligand-receptor data
        xy = np.array(adata.obsm['spatial']).copy()
        ligX = adata.to_df(layer=self.layer)[self.ligands].values
        recpX = adata.to_df(layer=self.layer)[self.receptors].values

        # Calculate weights using gaussian kernel
        lr_exp = []
        for i in range(len(xy)):
            w = gaussian_kernel_2d(xy[i], xy, radius=self.radius)

            ligand_exp = (ligX.T * w).T
            receptor_exp = recpX[i]
            lr_exp.append((ligand_exp * receptor_exp).mean(axis=0))

        lr_exp = torch.from_numpy(np.stack(lr_exp, axis=0)).float()

        X = torch.cat([X, lr_exp], dim=1)

        if cache_exists and cache:
            with open(beta_dists_file, 'rb') as f:
                self.beta_dists = pickle.load(f)

        else:
            # self.beta_model = BayesianRegression(
            #     n_regulators=len(self.regulators)+len(self.ligands), 
            #     device=torch.device('cpu') ## use cpu for better parallelization
            # )

            # ns = 1 if self.test_mode else num_samples
            # _max_epochs = 1 if self.test_mode else 3000


            # self.beta_model.fit(
            #     X, y, cluster_labels, 
            #     max_epochs=_max_epochs, learning_rate=3e-3, 
            #     num_samples=ns,
            #     parallel=parallel
            # )


            self.beta_dists = {}
            # for cluster in range(self.n_clusters):
            #     self.beta_dists[cluster] = self.beta_model.get_betas(
            #         X[cluster_labels==cluster].to(self.beta_model.device), 
            #         cluster=cluster, 
            #         num_samples=ns
            #     )
        
            for cluster in range(self.n_clusters):
                m = BayesianRidge()
                m.fit(X, y)
                y_pred = m.predict(X)
                r2 = r2_score(y, y_pred)
                print(f'cluster {cluster}: R2: {r2}')
                self.beta_dists[cluster] = np.hstack([m.intercept_, m.coef_]).reshape(1, -1)


            with open(beta_dists_file, 'wb') as f:
                pickle.dump(self.beta_dists, f)

        

        self.is_real = pd.DataFrame(
            [self.test_significance(self.beta_dists[i][:, 1:], alpha=alpha) for i in self.beta_dists.keys()], 
            columns=list(self.regulators)+list(self.lr['pairs'])
        ).T

        # for c in self.is_real.columns:
        #     for ix, s in enumerate(self.is_real[c].values):
        #         if not s:
        #             self.beta_dists[c][:, ix+1] = 0


        del X, y, cluster_labels

        try:
            _max_epochs = 1 if self.test_mode else max_epochs

            model, losses = self._build_model(
                adata,
                annot,
                spatial_dim=spatial_dim, 
                mode=mode,
                layer=self.layer,
                max_epochs=_max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rotate_maps=rotate_maps,
                pbar=pbar
            )
            
            self.model = model  
            self.losses = losses
        
        except KeyboardInterrupt:
            print('Training interrupted...')


            
    @torch.no_grad()
    def get_betas(self, xy=None, spatial_maps=None, labels=None, spatial_dim=None, beta_dists=None):

        assert xy is not None or spatial_maps is not None
        assert beta_dists is not None or self.beta_dists is not None
        spatial_dim = self.spatial_dim if spatial_dim is None else spatial_dim
        
        if spatial_maps is None:
            spatial_maps = xyc2spatial_fast(
                xyc = np.column_stack([xy, labels]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            ).astype(np.float32)

        if beta_dists is None:
            beta_dists = self.beta_dists
        
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
            anchors = np.stack(
                [beta_dists[label].mean(0) for label in batch_labels.cpu().numpy()], 
                axis=0
            )
            
            betas = self.model(batch_spatial.to(device), batch_labels.to(device))
            beta_list.extend(anchors*betas.cpu().numpy())
        
        return np.array(beta_list)
    

    def export(self):
        self.model.eval()

        return (
            self.model, 
            self.beta_dists, 
            self.is_real, 
            self.regulators, 
            self.target_gene
        )
    
    @property
    def betadata(self):
        betas = self.get_betas(
            spatial_maps=np.array(self.adata.obsm['spatial_maps']),
            labels=np.array(self.adata.obs[self.annot]))
        
        betas_df = pd.DataFrame(
            betas, 
            columns=['beta0']+['beta_'+i for i in self.is_real.index], 
            index=self.adata.obs.index
        )
        
        gex_df = self.adata.to_df(layer=self.layer)
        received_ligands = self.adata.uns['received_ligands']



        ## wL is the amount of ligand 'received' at each location
        ## assuming ligands and receptors expression are independent, dL/dR = 0
        ## y = b0 + b1*TF1 + b2*wL1R1 + b3*wL1R2
        ## dy/dTF1 = b1
        ## dy/dwL1 = b2[wL1*dR1/dwL1 + R1] + b3[wL1*dR2/dwL1 + R2]
        ##         = b2*R1 + b3*R2
        ## dy/dR1 = b2*[wL1 + R1*dwL1/dR1] = b2*wL1


        b_ligand = lambda x, y: betas_df[f'beta_{x}${y}']*received_ligands[x]
        b_receptor = lambda x, y: betas_df[f'beta_{x}${y}']*gex_df[y]

        ## dy/dR
        ligand_betas = pd.DataFrame(
            [b_ligand(x, y).values for x, y in zip(self.ligands, self.receptors)],
            columns=self.adata.obs.index, index=['beta_'+k for k in self.ligands]).T
        
        ## dy/dwL
        receptor_betas = pd.DataFrame(
            [b_receptor(x, y).values for x, y in zip(self.ligands, self.receptors)],
            columns=self.adata.obs.index, index=['beta_'+k for k in self.receptors]).T
        
        ## linearly combine betas for the same ligands or receptors
        ligand_betas = ligand_betas.groupby(lambda x:x, axis=1).sum()
        receptor_betas = receptor_betas.groupby(lambda x: x, axis=1).sum()

        assert not any(ligand_betas.columns.duplicated())
        assert not any(receptor_betas.columns.duplicated())
        
        xy = pd.DataFrame(self.adata.obsm['spatial'], index=self.adata.obs.index, columns=['x', 'y'])
        gex_modulators = self.regulators+self.ligands+self.receptors+[self.target_gene]
        

        """
        # Combine all relevant data into a single DataFrame
        # one row per cell
        betas_df \                                      # beta coefficients, TFs and LR-pairs
            .join(gex_df[np.unique(gex_modulators)]) \  # gene expression data for each modulator
            .join(self.adata.uns['ligand_receptor']) \  # weighted-ligands*receptor values
            .join(ligand_betas) \                       # beta_wLR * wL, 
            .join(receptor_betas) \                     # beta_wLR * R
            .join(self.adata.obs) \                     # cell type metadata
            .join(xy)                                   # spatial coordinates

        """

        
        return betas_df \
            .join(gex_df[np.unique(gex_modulators)]) \
            .join(self.adata.uns['ligand_receptor']) \
            .join(ligand_betas) \
            .join(receptor_betas) \
            .join(self.adata.obs) \
            .join(xy)


