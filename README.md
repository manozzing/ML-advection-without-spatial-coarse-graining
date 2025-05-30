# ML-advection-without-spatial-coarse-graining
This is a repository of machine-learned advection solver to accelerate the transport operator in chemical transport model. 

“data” directory contains model parameters and coastline for visualization. Training/testing data will be uploaded in a separate zenodo.
“src” contains utility functions.
- Flux_to_SC_params.jl: function codes to define the structure of learned solver in SimpleChains.jl and transfer the Flux.jl model parameters to SimpleChains.jl 
- advection_operators.jl: functions to run 1-D and 2-D advection using the learned solver
- training.jl: training functions
- vizualization_utils.jl: utility functions for map visualization
"test" directory has separate Julia files to run 2-D testing and generalization tests.
