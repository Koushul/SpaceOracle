from abc import ABC, abstractmethod

class Oracle(ABC):
    
    def __init__(self, adata):
        self.adata = adata
        
    
    @abstractmethod
    def _estimate_mean_betas(self, adata):
        pass
    
    
    
    
    
class SpaceOracle(Oracle):
    
    def __init__(self, adata, n_perturbations=100):
        self.adata = adata
        self.mean_betas = self._estimate_mean_betas(adata)
        
    def _estimate_mean_betas(self, adata):
        return adata.X.mean(axis=0)
    
    def perturb(self, adata, tf):
        pass
    
    

