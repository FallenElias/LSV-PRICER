import numpy as np
from typing import Tuple, Callable, Dict
from scipy.interpolate import RectBivariateSpline


def simulate_heston(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
    q: float,
    maturities: np.ndarray,
    n_steps: int,
    n_paths: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate (S_t, v_t) under the Heston dynamics via Euler‐Maruyama.

    Returns
    -------
    S_paths : array, shape (n_paths, n_steps+1)
    v_paths : array, shape (n_paths, n_steps+1)
    """
    dt = maturities[-1] / n_steps
    S = np.full((n_paths, n_steps+1), S0)
    v = np.full((n_paths, n_steps+1), v0)
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        z1 = np.random.normal(size=n_paths)
        z2 = rho*z1 + np.sqrt(1-rho**2)*np.random.normal(size=n_paths)
        v_prev = v[:, t]
        # full‐truncation Euler for v
        v_next = np.maximum(v_prev + kappa*(theta - np.maximum(v_prev,0))*dt
                            + xi*np.sqrt(np.maximum(v_prev,0))*sqrt_dt*z2, 0)
        S[:, t+1] = S[:, t]*np.exp((r - q - 0.5*v_prev)*dt + np.sqrt(v_prev*dt)*z1)
        v[:, t+1] = v_next
    return S, v


def estimate_conditional_variance(
    S_paths: np.ndarray,
    v_paths: np.ndarray,
    K_grid: np.ndarray,
    T_index: int,
    bandwidth: float = None,
) -> np.ndarray:
    """
    At time index T_index, bin paths by terminal S_T into bins around each K in K_grid
    and compute E[v_T | S_T≈K].

    Returns
    -------
    cond_var : array, shape (len(K_grid),)
    """
    ST = S_paths[:, T_index]
    vT = v_paths[:, T_index]
    cond_var = np.zeros(K_grid.shape, dtype=float)
    if bandwidth is None:
        bandwidth = (K_grid.max() - K_grid.min()) / len(K_grid)
    for i, K in enumerate(K_grid):
        mask = np.abs(ST - K) <= bandwidth
        if np.any(mask):
            cond_var[i] = np.mean(vT[mask])
        else:
            cond_var[i] = np.nan
    return cond_var


def build_leverage_function(
    strikes: np.ndarray,
    maturities: np.ndarray,
    local_vol: np.ndarray,
    cond_var: np.ndarray,
) -> Callable[[float, float], float]:
    """
    Build L(K,T) = σ_loc(K,T) / sqrt(cond_var(K,T)).

    If the grid is too small to fit a spline (e.g. 1×1), return a constant L.
    """
    # compute raw leverage surface
    Lgrid = np.where(cond_var > 0, local_vol / np.sqrt(cond_var), np.nan)

    # if only one valid point, return constant function
    if strikes.size < 2 or maturities.size < 2:
        # find first non-nan
        idx = np.argwhere(~np.isnan(Lgrid))
        if idx.size == 0:
            constL = np.nan
        else:
            i, j = idx[0]
            constL = float(Lgrid[i, j])
        return lambda K, T: constL

    # mask NaNs
    mask = ~np.isnan(Lgrid)
    # valid maturities/strikes
    T_valid = maturities[np.any(mask, axis=1)]
    K_valid = strikes[np.any(mask, axis=0)]
    Z = Lgrid[np.ix_(np.any(mask, axis=1), np.any(mask, axis=0))]

    # bilinear spline
    spline = RectBivariateSpline(T_valid, K_valid, Z, kx=1, ky=1)

    return lambda K, T: float(spline(T, K))