import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Callable, Tuple, Dict


def fit_iv_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_matrix: np.ndarray,
    kx: int = 3,
    ky: int = 3,
) -> Callable[[float, float], float]:
    """
    Fit a smooth implied-volatility surface σ_imp(K,T) via 2D spline.

    Parameters
    ----------
    strikes : np.ndarray, shape (M,)
        Sorted array of strikes K.
    maturities : np.ndarray, shape (N,)
        Sorted array of maturities T (in years or days).
    iv_matrix : np.ndarray, shape (N, M)
        Matrix of implied vols, rows correspond to maturities, columns to strikes.
    kx : int
        Degree of spline in K (default cubic).
    ky : int
        Degree of spline in T (default cubic).

    Returns
    -------
    iv_func : Callable[[float, float], float]
        Function iv_func(K, T) that returns interpolated implied vol.
    """
    # ensure inputs are sorted
    Ks = np.asarray(strikes)
    Ts = np.asarray(maturities)
    IV = np.asarray(iv_matrix)
    # build the spline
    spline = RectBivariateSpline(Ts, Ks, IV, kx=kx, ky=ky)
    def iv_func(K: float, T: float) -> float:
        return float(spline(T, K))
    return iv_func


def svi_parametrize(
    log_moneyness: np.ndarray,
    total_var: np.ndarray,
    initial_guess: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Calibrate SVI parameters (a, b, rho, m, sigma) to total implied variance.

    SVI total variance: w(k) = a + b [ rho (k − m) + sqrt((k − m)^2 + σ^2) ]

    Parameters
    ----------
    log_moneyness : np.ndarray, shape (L,)
        Array of log-strike ratios k = ln(K/F).
    total_var : np.ndarray, shape (L,)
        Corresponding total variance w = σ_imp(k)^2 T.
    initial_guess : dict, optional
        Initial parameter guess: {'a':..., 'b':..., 'rho':..., 'm':..., 'sigma':...}

    Returns
    -------
    params : dict
        Calibrated SVI parameters {'a','b','rho','m','sigma'}.
    """
    # stub: user to implement minimization over objective
    raise NotImplementedError("SVI calibration not yet implemented")
