from scipy.stats import pearsonr

from .fixtures.simulator import SimulatedData

class SimulationBetaCallback:
        
    def __call__(self, betas):
        assert betas.shape == SimulatedData.beta_shape
        return {
            'beta0': pearsonr(betas[:, 0], SimulatedData.beta0).statistic,
            'beta1': pearsonr(betas[:, 1], SimulatedData.beta1).statistic,
            'beta2': pearsonr(betas[:, 2], SimulatedData.beta2).statistic
        }
    
