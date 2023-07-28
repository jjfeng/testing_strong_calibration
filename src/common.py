import os
import numpy as np

def to_safe_logit(prob, eps=1e-10):
    prob_aug = np.maximum(eps, np.minimum(1 - eps, prob))
    return np.log(prob_aug/(1 - prob_aug))

def make_prob(raw_prob):
    return np.maximum(0, np.minimum(1, raw_prob))

def get_n_jobs():
    n_cpu = int(os.getenv('OMP_NUM_THREADS')) if os.getenv('OMP_NUM_THREADS') is not None else 0
    n_jobs = max(n_cpu - 1, 1) if n_cpu > 0 else -1
    return n_jobs
