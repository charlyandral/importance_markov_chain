import numpy as np
import numpy.random as npr
from numba import njit


@njit()
def ESS_IS(weights: np.ndarray) -> float:
    """Compute the importance sampling effective sample size.

    Args:
        weights (np.ndarray): weights of the IS sample

    Returns:
        float: Value of the ESS
    """

    ess = weights.sum(axis=0) ** 2 / (weights**2).sum(axis=0)
    return ess


@njit()
def compute_chain(
    points: np.ndarray, weights: np.ndarray, alpha: float = 1, kind: str = "bernoulli"
) -> tuple[np.array, np.array]:
    """Compute a Markov chain from a weighted sample. Spectify the algorithm (OSR or IMC with shifted bernoulli number of replicas) with the kind argument.

    Args:
        points (np.ndarray): points of the sample
        weights (np.ndarray): corresponding weights
        alpha (float, optional): tuning parameter for IMC. Defaults to 1.
        kind (str, optional): Algorithm to use, either OSR or IMC with shifted Bernoulli replicas. Defaults to "bernoulli".

    Raises:
        ValueError: raised if kind not in ["osr", "bernoulli"]

    Returns:
        tuple[np.array,np.array]: first element is the chain, second element is the number of copies of each point (corresponding to the elements in "points")
    """

    n_iter = points.shape[0]
    copies = np.ones(n_iter, dtype=np.int_)
    ratios = weights
    kappa = alpha * n_iter / ratios.sum()
    # compute copies in case of OSR
    if kind == "osr":
        for i in range(n_iter):
            if ratios[i] == 0:
                copies[i] = 0
            else:
                q = min(1, kappa * ratios[i])
                alpha = min(1 / (kappa * ratios[i]), 1)
                copies[i] = npr.geometric(alpha) * (npr.random() < q)
    # compute copies in case of IMC with shifted Bernoulli replicas
    elif kind == "bernoulli":
        for i in range(n_iter):
            copies[i] = int(kappa * ratios[i]) + (npr.rand() < (kappa * ratios[i] % 1))
    else:
        raise ValueError("kind must be 'stratified', 'osr' or 'bernoulli'")
    # from the points and the number of copies, compute the chain
    n_tot = np.cumsum(copies)
    n_tot = np.concatenate((np.zeros(1, dtype=np.int_), n_tot))
    out = np.zeros((n_tot[-1], points.shape[1]))
    for i in range(0, n_iter):
        out[n_tot[i] : n_tot[i + 1]] = points[i]
    return out, copies


def JMP(chain: np.ndarray) -> float:
    """Compute the average of the square of the jumps of a chain.

    Args:
        chain (np.ndarray): chain to compute the average jump of

    Returns:
        float: average quadratic jump
    """
    u = np.diff(chain, axis=0)
    out = np.mean(u**2)
    return out


def indep_MH_nf(
    proposals: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Markov chain from a weighted sample using independent MH steps.

    Args:
        proposals (np.ndarray): proposals of the MH steps
        weights (np.ndarray): corresponding weights

    Returns:
        tuple[np.ndarray,np.ndarray]: first element is the chain, second element is the number of copies of each point (corresponding to the elements in "points")
    """
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
def seed_numba(seed: int) -> None:
    """Set the seed for the numba random number generator.

    Args:
        seed (int): Value of the seed
    """
    npr.seed(seed)
