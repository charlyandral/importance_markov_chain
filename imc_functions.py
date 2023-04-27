
import numpy as np
import numpy.random as npr
from jax.config import config
from numba import njit



#config.update("jax_enable_x64", True)  # ajout des floats64




@njit()
def ESS_IS(copies):
    ess_kappa = copies.sum(axis=0) ** 2 / (copies**2).sum(axis=0)
    return ess_kappa








@njit()
def compute_chain(
    points: np.array, weights: np.array, alpha: float = 1, kind: str = "bernoulli"
):
    n_iter = points.shape[0]
    copies = np.ones(n_iter, dtype=np.int_)
    ratios = weights
    kappa = alpha * n_iter / ratios.sum()
    if kind == "stratified":
        u = npr.rand()
        for i in range(n_iter):
            u += (kappa * ratios[i]) % 1
            copies[i] = int(kappa * ratios[i]) + (u > 1)
            u = u % 1
    elif kind == "osr":
        for i in range(n_iter):
            if ratios[i] == 0:
                copies[i] = 0
            else:                    
                q = min(1, kappa * ratios[i])
                alpha = min(1 / (kappa * ratios[i]), 1)
                copies[i] = npr.geometric(alpha) * (npr.random() < q)
    elif kind == "bernoulli":
        for i in range(n_iter):
            copies[i] = int(kappa * ratios[i]) + (npr.rand() < (kappa * ratios[i] % 1))
    else:
        raise ValueError("kind must be 'stratified', 'osr' or 'bernoulli'")
    n_tot = np.cumsum(copies)
    n_tot = np.concatenate((np.zeros(1, dtype=np.int_), n_tot))
    out = np.zeros((n_tot[-1], points.shape[1]))
    for i in range(0, n_iter):
        out[n_tot[i] : n_tot[i + 1]] = points[i]
    return out, copies


# @nb.njit()
def JMP(chain):
    u = np.diff(chain, axis=0)
    out = np.mean(u**2)
    return out


def indep_MH_nf(proposals, weights):
    n_iter = proposals.shape[0]
    out = np.zeros_like(proposals)
    out[0] = proposals[0]
    accepted = 0
    copies = np.zeros((n_iter))
    pos_copie = 0
    current_weight = weights[0]
    for i in range(1, n_iter):
        alpha = min(weights[i] / current_weight, 1)
        if npr.random() < alpha:
            out[i] = proposals[i]
            accepted += 1
            pos_copie += 1
            current_weight = weights[i]
            copies[pos_copie] += 1
        else:
            out[i] = out[i - 1]
            copies[pos_copie] += 1
    return out, copies


@njit()
def seed_numba(seed):
    npr.seed(seed)


