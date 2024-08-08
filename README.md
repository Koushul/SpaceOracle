# SpaceOracle

Apple Silicon requires clang for velocyto:

```{pytohn}
brew install llvm libomp
export CC=/opt/homebrew/opt/llvm/bin/clang
```

Example: test a model on simulated data

```{python}
from spaceoracle.callbacks.simulation_callback import SimulationBetaCallback
from spaceoracle.callbacks.fixtures.simulator import SimulatedData
from spaceoracle.models.estimators import GeoCNNEstimator
```

```{python}
estimator = GeoCNNEstimator()

estimator.fit(
    SimulatedData.X, 
    SimulatedData.y, 
    SimulatedData.xy, 
    labels = SimulatedData.clusters,
    init_betas='ones', 
    max_epochs=5, 
    learning_rate=3e-3, 
    spatial_dim=8,
    in_channels=1,
    mode = 'train',
)
```

```{python}
check_betas(
    estimator.get_betas(
        SimulatedData.X, 
        SimulatedData.xy, 
        SimulatedData.clusters
    )[0]
)
```
