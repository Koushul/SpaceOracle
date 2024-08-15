import pandas as pd
import os
import numpy as np
from dataclasses import dataclass, field

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
    """
    ncells: int = 1000
    ntfs: int = 19  # Default number of transcription factors
    clusters: int = 7  # Default number of clusters

    labels: list = field(init=False)
    betas: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    xy: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self):
        cell_pos = [self.generate_positions(r, r + 1) for r in range(self.clusters)]
        coords = np.vstack([np.array(pos) for pos in cell_pos])
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        self.xy = np.hstack((x_coords[:, None], y_coords[:, None]))
        self.labels = np.array([r for r in range(self.clusters) for _ in range(self.ncells)])

        tfs = np.array(list(range(self.ntfs)))

        self.betas, self.y, self.X = self.beta_func(x_coords, y_coords, self.labels, tfs) # the y and X are switched on purpose!

    def beta_func(self, x, y, c, tf_index):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        
        tf_gexes = []
        for label in c:
            tf_ex = [label * (tf+1) % 3 * 0.1 for tf in range(self.ntfs)]
            tf_gexes.append(np.array(tf_ex))
        tf_gexes = np.array(tf_gexes)

        c = np.array(c).flatten()

        betas = (c + 1) * (x + y) / (c%2 + 1)
        betas = betas * (tf_index[:, None] + 1)
        coeff = np.linspace(-5, 1, num=x.shape[0]) 

        X = betas + coeff  
        X = X.sum(axis = 0)

        betas = np.hstack((coeff[:, None], betas.T))

        return betas, X, tf_gexes


    def generate_positions(self, radius_min, radius_max):
        # Generate positions for cells
        positions = []
        for _ in range(self.ncells):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(radius_min, radius_max)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append([x, y])
        return positions