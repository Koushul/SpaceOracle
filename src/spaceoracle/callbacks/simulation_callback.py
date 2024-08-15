from scipy.stats import pearsonr
import pickle

from .fixtures.simulator import SimulatedData, SimulatedDataV2

class SimulationBetaCallback:
        
    def __call__(self, betas):
        assert betas.shape == SimulatedData.beta_shape
        return {
            'beta0': pearsonr(betas[:, 0], SimulatedData.beta0).statistic,
            'beta1': pearsonr(betas[:, 1], SimulatedData.beta1).statistic,
            'beta2': pearsonr(betas[:, 2], SimulatedData.beta2).statistic
        }
    


class SimulationBetaCallbackV2:
        
    def __call__(self, beta_pred, beta_truth):
        assert beta_pred.shape == beta_truth.shape
        return {
            f'beta{i}': pearsonr(beta_pred[:, i], beta_truth[:,i]).statistic
            for i in range(beta_pred.shape[1])
        }
