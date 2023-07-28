import numpy as np

X_RANGE = 5

def create_class_dataset(dataset_type: str, n_obs: int, orig_beta: np.ndarray, new_beta: np.ndarray = None, orig_intercept: float = 0, new_intercept: float = 0):
    x = np.random.uniform(low=-X_RANGE, high=X_RANGE, size=(n_obs, orig_beta.size))
    num_p = orig_beta.size
    orig_mu = 1 / (1 + np.exp(-x @ orig_beta - orig_intercept))
    if dataset_type == "simple":
        mu = orig_mu
    else:
        true_grouping = np.ones((n_obs, 1))
        if dataset_type == "replace":
            true_grouping[:] = 0
        elif dataset_type == "subgroup":
            true_grouping[(x[:,0] < 0) & (x[:,1]<0)] = 0
        elif dataset_type == "mini":
            true_grouping[(x[:,0] < -2) & (x[:,1] > 2)] = 0
        else:
            raise NotImplementedError("unknown dataset type")
        new_mu =  1 / (1 + np.exp(-x @ new_beta - new_intercept))
        mu = orig_mu * true_grouping + new_mu * (1 - true_grouping)
        print("grouping shift amount", (true_grouping == 0).mean())
    
    y = np.random.binomial(1, mu.flatten(), size=n_obs).reshape((-1, 1))
    return (x, mu, y)
