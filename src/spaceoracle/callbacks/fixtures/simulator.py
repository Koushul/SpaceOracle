import pandas as pd
import os
import numpy as np
from dataclasses import dataclass, field
import anndata


@dataclass
class SimulatedData:
    """
    Holds simulated data
    `gt` - ground truth full data
    `X` - dependent variables
    `y` - independent variable
    `xy` - spatial xy coordinates
    `clusters` - cluster labels
    `beta0` - linear intercept
    `beta1` - linear coefficient 1
    `beta2` - linear coefficient 2
    """
    
    gt = pd.read_csv(
        os.path.dirname(
            os.path.realpath(__file__))+'/simulated_sinusoidals.csv')
    
    X = gt[['x1', 'x2']].values
    y = gt['y'].values
    xy = gt[['latitude', 'longitude']].values
    clusters  = np.zeros((len(y), 1))
    
    beta0 = gt['beta0'].values
    beta1 = gt['beta1'].values
    beta2 = gt['beta2'].values
    
    beta_shape = (len(gt), 3)

@dataclass
class SimulatedDataV2:
    """
    Holds simulated data based on CO data
    `X` - tf gexes
    `y` - target gex
    `xy` - spatial xy coordinates
    `clusters` - nclusters
    `labels` - cell cluster label
    """
    density: float = 0.3
    ntfs: int = 19  
    clusters: int = 7 
    position: str = 'circle'

    labels: list = field(init=False)
    betas: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    xy: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    adata: anndata.AnnData = field(init=False)

    def __post_init__(self):
        self.tf_labels = [f'tf_{i+1}' for i in range(self.ntfs)]
        
        cell_pos, ncells = zip(*[self.generate_positions(r, r+1) for r in range(self.clusters)])
        coords = np.vstack([np.array(pos) for pos in cell_pos])
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        self.xy = np.hstack((x_coords[:, None], y_coords[:, None]))
        self.labels = np.repeat(np.arange(len(ncells)), ncells)

        self.betas, self.X = self.beta_func(x_coords, y_coords, self.labels)
        self.betas, self.y, self.X = self.set_targex()
        self.adata = self.package_adata()

    def beta_func(self, x, y, c):
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        tfs = np.random.rand(self.ntfs)
        
        tf_gexes = []
        for tf in tfs:
            tf_ex = np.array([abs((c%5) + (np.sin(tf+1) + np.cos(x+y)))])
            tf_gexes.append(np.array(tf_ex))                # 1, cell
        tf_gexes = np.squeeze(np.array(tf_gexes)).T         # cell, TF

        betas = [(np.random.rand(x.shape[0]) * 0.8)]        # beta_0 coeff
        
        cmap_dict = {c: np.random.rand() for c in range(self.clusters)}
        replacements = np.vectorize(lambda x: cmap_dict.get(x, x))
        cmap = replacements(c)
        
        noise = 2           # increase to cause more variation within cluster
        beta_var = 10       # increase to cause greater variation between clusters
        beta_mag = 0.1      # increase to get bigger beta magnitude

        for tf in tfs:                                      # generate betas for each tf
            beta = np.array([beta_var*cmap + noise*(np.cos(noise+x)+np.sin(noise+y))]) * (beta_mag*tf)
            betas.append(beta.squeeze())     
        betas = np.array(betas).squeeze().T                 # beta_tf, cell

        return betas, tf_gexes

    def set_targex(self):
        betas = self.betas
        tf_gexes = self.X
        
        X = np.array([(betas[i, 1:] * tf_gex + betas[i, 0]) for i, tf_gex in enumerate(tf_gexes)])
        X = np.sum(X, axis = 1)                             # cell, 1

        scale_down = max(np.max(tf_gexes), np.max(X))
        tf_gexes /= scale_down
        X /= scale_down

        betas[:, 0] /= scale_down

        return betas, X, tf_gexes

    def generate_positions(self, radius_min, radius_max):
        positions = []

        area = np.pi*radius_max**2 - np.pi*radius_min**2
        ncells = round(area * self.density)
        
        for _ in range(ncells):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(radius_min, radius_max)
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if self.position == 'wave':
                x = 0.8 * abs(x) * np.random.uniform(1, 2.5)
                y = abs(y) * np.random.uniform(1, 2)
            positions.append([x, y])

        return positions, ncells
    
    def package_adata(self):
        df = pd.DataFrame(self.X)
        df.columns = self.tf_labels
        df['target_gene'] = self.y
        
        adata = anndata.AnnData(df)

        adata.obs['sim_cluster'] = self.labels
        adata.obsm['spatial'] = self.xy
        return adata