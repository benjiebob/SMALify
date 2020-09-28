from global_utils import config
import numpy as np
import pickle as pkl

def load_shape_prior(shape_family, use_smal_shape_prior):
    if use_smal_shape_prior:
        with open(config.SMAL_DATA_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        # Legacy code to enable 
        if shape_family == -1:
            model_covs = np.array(smal_data['cluster_cov'])[0]
            model_mean = np.array(smal_data['cluster_means'])[0]
            # TODO: and be careful because the means are 0 for these betas!
            model_init = np.array(smal_data['toys']) # TODO
        else:
            model_covs = np.array(smal_data['cluster_cov'])[shape_family]
            model_mean = smal_data['cluster_means'][shape_family]
            # TODO: and be careful because the means are 0 for these betas!
            model_init = np.array(smal_data['toys'][config.SMAL_DOG_TOY_IDS])
    else:
        unity_data = np.load(config.UNITY_SHAPE_PRIOR) 
        model_covs = unity_data['cov'][:-1, :-1]
        model_mean = unity_data['mean'][:-1]
        model_init = unity_data['models'][:, :-1] # hack for now

    model_covs = model_covs + 1e-5 * np.eye(model_covs.shape[0])
    return model_mean, model_covs, model_init