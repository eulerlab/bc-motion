SAC & BC Model
- This folder runs separately and does not require a .yaml file
- Each notebook contains one model and saves DSI values at the end of the notebook
- Some notebooks contain code for making figures, indicated by the name of the file
- For comparisons across models, the notebooks load DSI values
- BC RFs in 'BC Data' are named after their cluster and likely corresponding type
- BC RFs in 'BC Data' with names including 'enhanced_X' are RFs with increased or decreased surround strength

Structure of notebooks
- Load where the BCs and their synapses are located along the SAC dendrite
- Load spatial and temporal resolution of the BC RFs
- Load BC RFs (kernels) and flip time axis (because of np convolve)
- Place BCs along the SAC dendrite specific to their location
- Run model 
- Extract DSI values and other quantities from Brian2 object
- Plot membrane potential in most distal compartment for all velocities
- Save DSI values
