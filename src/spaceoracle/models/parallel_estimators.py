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
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import ARDRegression
from group_lasso import GroupLasso
from spaceoracle.models.spatial_map import xyc2spatial_fast
from spaceoracle.tools.network import DayThreeRegulatoryNetwork, expand_paired_interactions
from .pixel_attention import CellularNicheNetwork
from ..tools.utils import gaussian_kernel_2d, min_max_df, set_seed
import commot as ct
from scipy.spatial.distance import cdist
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

def received_ligands(xy, lig_df, radius=200, scale_factor=1e5):
    ligands = lig_df.columns
    gauss_weights = [
        scale_factor * gaussian_kernel_2d(
            xy[i], 
            xy, 
            radius=radius) for i in range(len(lig_df))
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
    # df = pd.DataFrame(StandardScaler().fit_transform(result), columns=columns, index=obs_index)
    df = pd.DataFrame(result, columns=columns, index=obs_index)
    
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


    def init_ligands_and_receptors(self, receptor_thresh=0.01):
        df_ligrec = ct.pp.ligand_receptor_database(
                database='CellChat', 
                species='mouse', 
                signaling_type="Secreted Signaling"
            )
            
        df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  

        self.lr = expand_paired_interactions(df_ligrec)
        self.lr = self.lr[self.lr.ligand.isin(self.adata.var_names) & (self.lr.receptor.isin(self.adata.var_names))]

        # receptors = self.lr['receptor']
        # recex_means = np.mean(self.adata.to_df()[receptors], axis=0)
        # self.lr = self.lr.iloc[np.argwhere(recex_means > receptor_thresh).flatten()]

        self.lr = self.lr[~((self.lr.receptor == self.target_gene) | (self.lr.ligand == self.target_gene))]
        self.lr['pairs'] = self.lr.ligand.values + '$' + self.lr.receptor.values
        self.lr = self.lr.drop_duplicates(subset='pairs', keep='first')
        self.ligands = list(self.lr.ligand.values)
        self.receptors = list(self.lr.receptor.values)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data', 'ligand_target_mouse.parquet'))
        nichenet_lt = pd.read_parquet(data_path)

        self.nichenet_lt = nichenet_lt.loc[
            np.intersect1d(nichenet_lt.index, self.regulators)][
                np.intersect1d(nichenet_lt.columns, self.ligands)]
        
        self.tfl_pairs = []
        self.tfl_regulators = []
        self.tfl_ligands = []

        self.ligand_regulators = {lig: set(
            self.grn.get_regulators(self.adata, lig)) for lig in self.nichenet_lt.columns}

        for tf_ in self.nichenet_lt.index:
            row = self.nichenet_lt.loc[tf_]
            top_5 = row.nlargest(5)
            for lig_, value in top_5.items():
                if self.target_gene not in self.ligand_regulators[lig_] and \
                    tf_ not in self.ligand_regulators[lig_] and \
                    value > self.tf_ligand_cutoff:
                    self.tfl_ligands.append(lig_)
                    self.tfl_regulators.append(tf_)
                    self.tfl_pairs.append(f"{lig_}#{tf_}")


        assert len(self.ligands) == len(self.receptors)
        assert len(self.tfl_regulators) == len(self.tfl_ligands)
        
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
            self.adata.uns['received_ligands'] = received_ligands(
                self.adata.obsm['spatial'], 
                self.adata.to_df(layer=self.layer)[np.unique(self.ligands)], 
                radius=self.radius,
            )

            self.adata.uns['received_ligands_tfl'] = received_ligands(
                self.adata.obsm['spatial'], 
                self.adata.to_df(layer=self.layer)[np.unique(self.tfl_ligands)], 
                radius=self.radius,
            )

            self.adata.uns['ligand_receptor'] = self.ligands_receptors_interactions(
                self.adata.uns['received_ligands'][self.ligands], 
                self.adata.to_df(layer=self.layer)[self.receptors]

            )

        else:
            self.adata.uns['received_ligands'] = pd.DataFrame(index=self.adata.obs.index)
            self.adata.uns['ligand_receptor'] = pd.DataFrame(index=self.adata.obs.index)


        if len(self.tfl_pairs) > 0:
            self.adata.uns['ligand_regulator'] = self.ligand_regulators_interactions(
                self.adata.uns['received_ligands_tfl'][self.tfl_ligands], 
                self.adata.to_df(layer=self.layer)[self.tfl_regulators]
            )
        else:
            self.adata.uns['ligand_regulator'] = pd.DataFrame(index=self.adata.obs.index)


        self.xy = np.array(self.adata.obsm['spatial'])
        cluster_labels = np.array(self.adata.obs[self.cluster_annot])

        self.xy_df = pd.DataFrame(self.xy, columns=['x', 'y'], index=self.adata.obs.index)

        self.spatial_maps = xyc2spatial_fast(
                xyc = np.column_stack([self.xy, cluster_labels]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            )
            
        self.adata.obsm['spatial_maps'] = self.spatial_maps


        self.train_df = self.adata.to_df(layer=self.layer)[
            [self.target_gene]+self.regulators] \
            .join(self.adata.uns['ligand_receptor']) \
            .join(self.adata.uns['ligand_regulator'])
        

        # self.train_df = min_max_df(self.train_df)

        self.spatial_features = create_spatial_features(
            self.adata.obsm['spatial'][:, 0], 
            self.adata.obsm['spatial'][:, 1], 
            self.adata.obs[self.cluster_annot], 
            self.adata.obs.index,
            radius=self.radius
        )

        self.adata.obsm['spatial_features'] = self.spatial_features.copy()

        self.spatial_features = pd.DataFrame(
            MinMaxScaler().fit_transform(self.spatial_features.values), 
            columns=self.spatial_features.columns, 
            index=self.spatial_features.index
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
            cluster_sp_maps = torch.from_numpy(
                self.sp_maps[mask][:, cluster_target:cluster_target+1, :, :]).float()
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
        betas_df = self.get_betas()

        xy = pd.DataFrame(
            self.adata.obsm['spatial'], 
            index=self.adata.obs.index, 
            columns=['x', 'y']
        )

        _data = betas_df \
            .join(self.adata.obs) \
            .join(xy)
        
        return _data


    def fit(self, num_epochs=10, threshold_lambda=1e-4, learning_rate=2e-4, batch_size=512, 
            use_ARD=False, pbar=None, discard=50):
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

        if num_epochs:
            print(f'Fitting {self.target_gene} with {len(self.modulators)} modulators')
            print(f'\t{len(self.regulators)} Transcription Factors')
            print(f'\t{len(self.lr_pairs)} Ligand-Receptor Pairs')
            print(f'\t{len(self.tfl_pairs)} TranscriptionFactor-Ligand Pairs')

        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            X_cell, y_cell = self.Xn[mask], self.yn[mask]

            if use_ARD: 

                X_tf = X_cell[:, :len(self.regulators)]
                X_lr = X_cell[:, len(self.regulators):len(self.regulators)+len(self.ligands)]
                X_tfl = X_cell[:, -len(self.tfl_pairs):]

                m1 = ARDRegression(threshold_lambda=threshold_lambda)
                m1.fit(X_tf, y_cell)

                m2 = ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=True)
                m2.fit(X_lr, y_cell)

                m3 = ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=True)
                m3.fit(X_tfl, y_cell)

                y_pred = (m1.predict(X_tf) + m2.predict(X_lr) + m3.predict(X_tfl)) / 3
                r2_ard = r2_score(y_cell, y_pred)

                intercept = (m1.intercept_ + m2.intercept_ + m3.intercept_) / 3
                coefs = np.hstack[m1.coef_, m2.coef_, m3.coef_]
                _betas = np.hstack([intercept, coefs])
                # _betas = np.hstack([intercept, m1.coef_, m2.coef_, m3.coef_])

                # m = ARDRegression(threshold_lambda=threshold_lambda)
                # m.fit(X_cell, y_cell)
                # y_pred = m.predict(X_cell)
                # r2_ard = r2_score(y_cell, y_pred)
                # _betas = np.hstack([m.intercept_, m.coef_])

                coefs = None
            
            else:

                groups = [1]*len(self.regulators) + [2]*len(self.ligands) + [3]*len(self.tfl_pairs)

                groups = np.array(groups)

                gl = GroupLasso(
                    groups=groups,
                    group_reg=threshold_lambda,
                    l1_reg=0,
                    frobenius_lipschitz=True,
                    scale_reg="inverse_group_size",
                    subsampling_scheme=1,
                    # supress_warning=True,
                    n_iter=1000,
                    tol=1e-3,
                )
                gl.fit(X_cell, y_cell)

                y_pred = gl.predict(X_cell)
                coefs = gl.coef_.flatten()

                def threshold_coefficients(coefs, group, discard=50):
                    '''higher discard % means we set higher threshold'''
                    group_coefs = coefs[groups == group]
                    if len(group_coefs) <= 0:
                        return []
                    thresh = np.percentile(abs(group_coefs), discard)
                    return np.where(abs(group_coefs) > thresh, group_coefs, 0)

                tf_coefs = threshold_coefficients(coefs, group=1, discard=discard)
                lr_coefs = threshold_coefficients(coefs, group=2, discard=discard)
                tfl_coefs = threshold_coefficients(coefs, group=3, discard=discard)

                _betas = np.hstack([gl.intercept_, tf_coefs, lr_coefs, tfl_coefs])

                r2_ard = r2_score(y_cell, y_pred)
            

            loader = DataLoader(
                RotatedTensorDataset(
                    sp_maps[mask],
                    X_cell,
                    y_cell,
                    cluster,
                    self.spatial_features.iloc[mask].values,
                    rotate_maps=False
                ),
                batch_size=batch_size, shuffle=True
            )

            assert _betas.shape[0] == len(self.modulators)+1

            model = CellularNicheNetwork(
                    n_modulators = len(self.modulators), 
                    anchors=_betas,
                    spatial_dim=self.spatial_dim,
                    n_clusters=self.n_clusters
                ).to(self.device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

            
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
                    loss += torch.mean(outputs.mean(0) - model.anchors) * 1e-4
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    all_y_true.extend(targets.cpu().detach().numpy())
                    all_y_pred.extend(outputs.cpu().detach().numpy())

                    pbar.desc = f'{self.target_gene} | {cluster+1}/{self.n_clusters}'
                    pbar.update(len(targets))

            if num_epochs:
                print(f'{cluster}: {r2_score(all_y_true, all_y_pred):.4f} | {r2_ard:.4f}')
            self.models[cluster] = model
