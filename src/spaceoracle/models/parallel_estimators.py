import copy
from anndata import AnnData
import enlighten
from sklearn.metrics import r2_score
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.linear_model import ARDRegression 
from spaceoracle.models.spatial_map import xyc2spatial_fast
from spaceoracle.tools.network import DayThreeRegulatoryNetwork, expand_paired_interactions
from .pixel_attention import CellularNicheNetwork
from ..tools.utils import gaussian_kernel_2d, min_max_df, set_seed
import commot as ct
import numba
set_seed(42)

@numba.njit(parallel=True)
def calculate_weighted_ligands(gauss_weights, lig_df_values, u_ligands):
    n_ligands = len(u_ligands)
    n_cells = len(gauss_weights)
    weighted_ligands = np.zeros((n_ligands, n_cells))
    
    for i in numba.prange(n_ligands):
        for j in range(n_cells):
            weighted_ligands[i, j] = np.mean(gauss_weights[j] * lig_df_values[:, i])
    
    return weighted_ligands


from scipy.spatial.distance import cdist

def create_spatial_features(x, y, celltypes, obs_index,radius=200):
    coords = np.column_stack((x, y))
    unique_celltypes = np.unique(celltypes)
    result = np.zeros((len(x), len(unique_celltypes)))
    distances = cdist(coords, coords)
    for i, celltype in enumerate(unique_celltypes):
        mask = celltypes == celltype
        neighbors = (distances <= radius)[:, mask]
        result[:, i] = np.sum(neighbors, axis=1)
    
    if result.shape != (len(x), len(unique_celltypes)):
        raise ValueError(f"Unexpected result shape: {result.shape}. Expected: {(len(x), len(unique_celltypes))}")
    
    columns = [f'{ct}_within' for ct in unique_celltypes]
    df = pd.DataFrame(StandardScaler().fit_transform(result), columns=columns, index=obs_index)
    
    return df


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RotatedTensorDataset(Dataset):
    def __init__(self, sp_maps, X_cell, y_cell, cluster, spatial_features, rotate_maps=True):
        self.sp_maps = sp_maps
        self.X_cell = X_cell
        self.y_cell = y_cell
        self.cluster = cluster
        self.spatial_features = spatial_features
        self.rotate_maps = rotate_maps

    def __len__(self):
        return len(self.X_cell)

    def __getitem__(self, idx):
        sp_map = self.sp_maps[idx, self.cluster:self.cluster+1, :, :]
        if self.rotate_maps:
            k = np.random.choice([0, 1, 2, 3])
            sp_map = np.rot90(sp_map, k=k, axes=(1, 2))


        return (
            torch.from_numpy(sp_map.copy()).float(),
            torch.from_numpy(self.X_cell[idx]).float(),
            torch.from_numpy(np.array(self.y_cell[idx])).float(),
            torch.from_numpy(self.spatial_features[idx]).float()
        )

class SpatialCellularProgramsEstimator:
    def __init__(self, adata, target_gene, spatial_dim=64, 
            cluster_annot='rctd_cluster', layer='imputed_count', 
            radius=200, tf_ligand_cutoff=0.01):
        assert isinstance(adata, AnnData), 'adata must be an AnnData object'
        assert target_gene in adata.var_names, f'{target_gene} must be in adata.var_names'
        assert layer in adata.layers, f'{layer} must be in adata.layers'
        assert cluster_annot in adata.obs.columns, f'{cluster_annot} must be in adata.obs.columns'

        
        self.adata = adata
        self.target_gene = target_gene
        self.cluster_annot = cluster_annot
        self.layer = layer
        self.device = device
        self.radius = radius
        self.spatial_dim = spatial_dim
        self.tf_ligand_cutoff = tf_ligand_cutoff
        self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        self.regulators = self.grn.get_cluster_regulators(self.adata, self.target_gene)

        self.init_ligands_and_receptors()
        self.lr_pairs = self.lr['pairs']
        
        self.n_clusters = len(self.adata.obs[self.cluster_annot].unique())
        self.modulators = self.regulators + list(self.lr_pairs) + self.tfl_pairs

        self.modulators_genes = list(np.unique(
            self.regulators+self.ligands+self.receptors+self.tfl_regulators+self.tfl_ligands))

        assert len(self.ligands) == len(self.receptors), 'ligands and receptors must have the same length for pairing'
        assert np.isin(self.ligands, self.adata.var_names).all(), 'all ligands must be in adata.var_names'
        assert np.isin(self.receptors, self.adata.var_names).all(), 'all receptors must be in adata.var_names'
        assert np.isin(self.regulators, self.adata.var_names).all(), 'all regulators must be in adata.var_names'


    def init_ligands_and_receptors(self):
        df_ligrec = ct.pp.ligand_receptor_database(
                database='CellChat', 
                species='mouse', 
                signaling_type="Secreted Signaling"
            )
            
        df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  

        self.lr = expand_paired_interactions(df_ligrec)
        self.lr = self.lr[self.lr.ligand.isin(self.adata.var_names) & (self.lr.receptor.isin(self.adata.var_names))]
        self.lr = self.lr[~((self.lr.receptor == self.target_gene) | (self.lr.ligand == self.target_gene))]
        self.lr['pairs'] = self.lr.ligand.values + '$' + self.lr.receptor.values
        self.lr = self.lr.drop_duplicates(subset='pairs', keep='first')
        self.ligands = list(self.lr.ligand.values)
        self.receptors = list(self.lr.receptor.values)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data', 'ligand_target.parquet'))
        nichenet_ligand_target = pd.read_parquet(data_path)
        nichenet_ligand_target = nichenet_ligand_target.loc[
            np.intersect1d(nichenet_ligand_target.index, self.regulators)][
                np.intersect1d(nichenet_ligand_target.columns, self.ligands)]
        

        # print(nichenet_ligand_target)
        
        self.tfl_pairs = []
        self.tfl_regulators = []
        self.tfl_ligands = []
        for idx, row in nichenet_ligand_target.iterrows():
            top_5 = row.nlargest(5)
            for col, value in top_5.items():
                if value > self.tf_ligand_cutoff:
                    self.tfl_ligands.append(col)
                    self.tfl_regulators.append(idx)
                    self.tfl_pairs.append(f"{col}#{idx}")


        assert len(self.ligands) == len(self.receptors)
        assert len(self.tfl_regulators) == len(self.tfl_ligands)

    def received_ligands(self, xy, lig_df, radius=200):
        ligands = lig_df.columns
        gauss_weights = [
            gaussian_kernel_2d(
                xy[i], 
                xy, 
                radius=radius) for i in range(len(lig_df)
            )
        ]

        u_ligands = list(np.unique(ligands))
        lig_df_values = lig_df[u_ligands].values
        weighted_ligands = calculate_weighted_ligands(
            gauss_weights, lig_df_values, u_ligands)

        return pd.DataFrame(
            weighted_ligands, 
            index=u_ligands, 
            columns=lig_df.index
        ).T
        
    @staticmethod
    def ligands_receptors_interactions(received_ligands_df, receptor_gex_df):
        assert isinstance(received_ligands_df, pd.DataFrame)
        assert isinstance(receptor_gex_df, pd.DataFrame)
        assert received_ligands_df.index.equals(receptor_gex_df.index)
        assert received_ligands_df.shape[1] == receptor_gex_df.shape[1]

        _received_ligands = received_ligands_df.values
        _self_receptor_expression = receptor_gex_df.values
        lr_interactions  = _received_ligands * _self_receptor_expression
        
        return pd.DataFrame(
            lr_interactions, 
            columns=[i[0]+'$'+i[1] for i in zip(
                received_ligands_df.columns, receptor_gex_df.columns)], 
            index=receptor_gex_df.index
        )
    
    @staticmethod
    def ligand_regulators_interactions(received_ligands_df, regulator_gex_df):
        assert isinstance(received_ligands_df, pd.DataFrame)
        assert isinstance(regulator_gex_df, pd.DataFrame)
        assert received_ligands_df.index.equals(regulator_gex_df.index)
        assert received_ligands_df.shape[1] == regulator_gex_df.shape[1]

        _received_ligands = received_ligands_df.values
        _self_regulator_expression = regulator_gex_df.values
        ltf_interactions  = _received_ligands * _self_regulator_expression
        
        return pd.DataFrame(
            ltf_interactions, 
            columns=[i[0]+'#'+i[1] for i in zip(
                received_ligands_df.columns, regulator_gex_df.columns)], 
            index=regulator_gex_df.index
        )


    def init_data(self):
        if len(self.lr['pairs']) > 0:
            self.adata.uns['received_ligands'] = self.received_ligands(
                self.adata.obsm['spatial'], 
                self.adata.to_df(layer=self.layer)[np.unique(self.ligands)], 
                radius=self.radius,
            )

            self.adata.uns['received_ligands_tfl'] = self.received_ligands(
                self.adata.obsm['spatial'], 
                self.adata.to_df(layer=self.layer)[np.unique(self.tfl_ligands)], 
                radius=self.radius,
            )

            self.adata.uns['ligand_receptor'] = self.ligands_receptors_interactions(
                self.adata.uns['received_ligands'][self.ligands], 
                self.adata.to_df(layer=self.layer)[self.receptors]
            )

            self.adata.uns['ligand_regulator'] = self.ligand_regulators_interactions(
                self.adata.uns['received_ligands_tfl'][self.tfl_ligands], 
                self.adata.to_df(layer=self.layer)[self.tfl_regulators]
            )
        else:
            self.adata.uns['received_ligands'] = pd.DataFrame(index=self.adata.obs.index)
            self.adata.uns['ligand_receptor'] = pd.DataFrame(index=self.adata.obs.index)
            self.adata.uns['ligand_regulator'] = pd.DataFrame(index=self.adata.obs.index)
        

        self.xy = np.array(self.adata.obsm['spatial'])
        cluster_labels = np.array(self.adata.obs[self.cluster_annot])


        self.spatial_maps = xyc2spatial_fast(
                xyc = np.column_stack([self.xy, cluster_labels]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            )
            
        self.adata.obsm['spatial_maps'] = self.spatial_maps

        self.train_df = self.adata.to_df(layer=self.layer)[
            self.regulators+[self.target_gene]] \
            .join(self.adata.uns['ligand_receptor']) \
            .join(self.adata.uns['ligand_regulator'])
        

        self.train_df = min_max_df(self.train_df)

        self.spatial_features = create_spatial_features(
            self.adata.obsm['spatial'][:, 0], 
            self.adata.obsm['spatial'][:, 1], 
            self.adata.obs[self.cluster_annot], 
            self.adata.obs.index,
            radius=self.radius
        )

        X = self.train_df.drop(columns=[self.target_gene]).values
        y = self.train_df[self.target_gene].values
        sp_maps = self.spatial_maps

        assert sp_maps.shape[0] == X.shape[0] == y.shape[0] == len(cluster_labels)
        
        return sp_maps, X, y, cluster_labels


    @torch.no_grad()
    def get_betas(self):
        index_tracker = []
        betas = []
        for cluster_target in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cluster_target
            indices = self.cell_indices[mask]
            index_tracker.extend(indices)
            cluster_sp_maps = torch.from_numpy(self.sp_maps[mask][:, cluster_target:cluster_target+1, :, :]).float()
            spf = torch.from_numpy(self.spatial_features.values[mask]).float()
            b = self.models[cluster_target].get_betas(
                cluster_sp_maps.to(self.device),
                spf.to(self.device)
            ).cpu().numpy()
            betas.extend(b)

        return pd.DataFrame(
            betas, 
            index=index_tracker, 
            columns=['beta0']+['beta_'+i for i in self.modulators]
        ).reindex(self.adata.obs.index)
    
    @property
    def betadata(self):
        gex_df = self.adata.to_df(layer=self.layer)
        received_ligands = self.adata.uns['received_ligands']

        # gex_df = pd.DataFrame(
        #     MinMaxScaler().fit_transform(gex_df),
        #     columns=gex_df.columns,
        #     index=gex_df.index
        # )

        gex_df = min_max_df(gex_df)

        if received_ligands.shape[1] > 0:
            # received_ligands = pd.DataFrame(
            #     MinMaxScaler().fit_transform(received_ligands),
            #     columns=received_ligands.columns,
            #     index=received_ligands.index
            # )
            received_ligands = min_max_df(received_ligands)
            

        betas_df = self.get_betas()

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
            columns=self.adata.obs.index, index=['beta_'+k for k in self.receptors]).T
        
        ## dy/dwL
        receptor_betas = pd.DataFrame(
            [b_receptor(x, y).values for x, y in zip(self.ligands, self.receptors)],
            columns=self.adata.obs.index, index=['beta_'+k for k in self.ligands]).T
        
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

        received_ligands.columns = ['received_'+i for i in received_ligands.columns]
        
        _data = betas_df \
            .join(gex_df[np.unique(gex_modulators)]) \
            .join(received_ligands) \
            .join(self.train_df[self.lr['pairs']]) \
            .join(ligand_betas) \
            .join(receptor_betas) \
            .join(self.adata.obs) \
            .join(xy)
        
        return _data


    def fit(self, num_epochs=10, threshold_lambda=1e4, learning_rate=2e-4, batch_size=512, pbar=None):
        sp_maps, X, y, cluster_labels = self.init_data()

        self.models = {}
        self.Xn = X
        self.yn = y
        self.sp_maps = sp_maps
        self.cell_indices = self.adata.obs.index
        self.cluster_labels = cluster_labels

        if pbar is None:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=sp_maps.shape[0]*num_epochs, 
                desc='Estimating Spatial Betas', unit='cells',
                color='green',
                auto_refresh=True
            )
        
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            X_cell, y_cell = self.Xn[mask], self.yn[mask]

            m = ARDRegression(threshold_lambda=threshold_lambda)
            m.fit(X_cell, y_cell)
            y_pred = m.predict(X_cell)
            r2_ard = r2_score(y_cell, y_pred)
            _betas = np.hstack([m.intercept_, m.coef_])


            loader = DataLoader(
                RotatedTensorDataset(
                    sp_maps[mask],
                    X_cell,
                    y_cell,
                    cluster,
                    self.spatial_features.iloc[mask].values,
                    rotate_maps=True
                ),
                batch_size=batch_size, shuffle=True
            )

            if not (_betas.shape[0] == len(self.modulators)+1):
                print(f'Mismatch for {self.target_gene} with {len(self.modulators)} modulators and {_betas.shape[0]} betas')
                print(X_cell.shape, self.train_df.shape)
                print(self.adata.uns['ligand_regulator'])

            model = CellularNicheNetwork(
                n_modulators = len(self.modulators), 
                anchors=_betas,
                spatial_dim=self.spatial_dim,
                n_clusters=self.n_clusters
            ).to(self.device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                all_y_true = []
                all_y_pred = []
                
                for batch in loader:
                    spatial_maps, inputs, targets, spatial_features = [b.to(device) for b in batch]
                    
                    optimizer.zero_grad()
                    outputs = model(spatial_maps, inputs, spatial_features)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    all_y_true.extend(targets.cpu().detach().numpy())
                    all_y_pred.extend(outputs.cpu().detach().numpy())
                    # pbar.desc = f'{cluster}: {r2_score(all_y_true, all_y_pred):.4f} | {r2_ard:.4f}'
                    pbar.desc = f'{self.target_gene} | {cluster+1}/{self.n_clusters}'
                    pbar.update(len(targets))

            print(f'{cluster}: {r2_score(all_y_true, all_y_pred):.4f} | {r2_ard:.4f}')
            self.models[cluster] = model
