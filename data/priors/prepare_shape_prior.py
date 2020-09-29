import numpy as np

unity_data = np.load("betas_shape_family_1_edit.npy")
unity_data[[2,7,11], 22] = -1.0
unity_data[[2,7,11], 23] = 0.0
# unity_data[:, 20:] = np.exp(unity_data[:, 20:])
        
model_covs = np.cov(unity_data, rowvar=False)
mean_betas_unimode = np.mean(unity_data, axis = 0)

np.savez_compressed('unity_betas.npz', 
    cov = model_covs,
    mean = mean_betas_unimode,
    models = unity_data)