from scipy.stats import pearsonr
import pickle
import copy

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

    def perturb(self, sim_data, tf_label):
        tf_index = sim_data.tf_labels.index(tf_label)
        
        psim_data = copy.deepcopy(sim_data)
        psim_data.X[:, tf_index] = 0
        
        psim_data.set_targex()
        psim_data.package_adata()
        
        return psim_data
        
        