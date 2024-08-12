import pandas as pd
import os
import numpy as np
from dataclasses import dataclass

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
    
   