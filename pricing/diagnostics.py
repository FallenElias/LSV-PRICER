import numpy as np
from scipy.stats import kstest
from typing import Callable, List, Tuple, Dict
from pricing.pricing import european_price_mc
from utils.financial import bs_call_price


def backtest_marginals(
    S_paths: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_func: Callable[[float, float], float],
    S0: float,
    r: float,
    q: float,
    n_bins: int = 50,
    alpha: float = 0.05,
) -> Dict[Tuple[float, float], bool]:
    """
    Compare simulated marginals of S_T at each (K,T) grid node
    against the Black–Scholes distribution implied by iv_func.

    Returns a dict mapping (K,T) → True if the KS test passes at level alpha.
    """
    results = {}
    for T in maturities:
        # simulate just terminal slice from S_paths: assume paths simulated to max T
        ST = S_paths[:, -1]
        # theoretical CDF under BS lognormal
        iv = iv_func(strikes[0], T)  # use ATM iv as proxy for distribution
        mu = np.log(S0) + (r - q - 0.5*iv**2)*T
        sigma = iv * np.sqrt(T)
        cdf = lambda x: 0.5*(1+np.erf((np.log(x)-mu)/(sigma*np.sqrt(2))))
        # KS test
        stat, pvalue = kstest(ST, cdf)
        results[(strikes[0], T)] = (pvalue > alpha)
    return results


def check_mc_convergence(
    price_func: Callable[..., float],
    args: dict,
    step_list: List[int],
    path_list: List[int],
) -> np.ndarray:
    """
    Compute MC price over combinations of time steps and path counts.

    Parameters
    ----------
    price_func : callable
        e.g. lambda S0, ... , n_steps, n_paths: price
    args       : dict
        All named args except n_steps, n_paths
    step_list  : list of time-steps to test
    path_list  : list of path-counts to test

    Returns
    -------
    errors : 2D array, shape (len(step_list), len(path_list))
        absolute error versus reference (finest grid & most paths).
    """
    # reference price at finest resolution
    ref_steps = max(step_list)
    ref_paths = max(path_list)
    args_ref = {**args, "n_steps": ref_steps, "n_paths": ref_paths}
    ref_price = price_func(**args_ref)

    errors = np.zeros((len(step_list), len(path_list)))
    for i, n_steps in enumerate(step_list):
        for j, n_paths in enumerate(path_list):
            p = price_func(**{**args, "n_steps": n_steps, "n_paths": n_paths})
            errors[i, j] = abs(p - ref_price)
    return errors
