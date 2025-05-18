import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Callable, Tuple, Dict


from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
import numpy as np

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def fit_iv_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_matrix: np.ndarray,
) -> Callable[[float, float], float]:
    """
    Fit an implied‐vol surface σ_imp(K,T) from scattered IV data
    via linear interpolation with nearest‐neighbor fallback.
    """
    # 1) collect all non‐NaN points
    Ks, Ts = np.meshgrid(strikes, maturities)
    mask   = ~np.isnan(iv_matrix)
    pts    = np.column_stack([Ks[mask], Ts[mask]])  # shape (n_pts, 2)
    vals   = iv_matrix[mask]                        # shape (n_pts,)

    if len(vals) == 0:
        raise ValueError("No IV data to fit surface")

    # 2) build a linear interpolator, and a nearest fallback
    lin_interp = LinearNDInterpolator(pts, vals)
    nn_interp  = NearestNDInterpolator(pts, vals)

    # 3) define the callable surface
    def iv_func(K: float, T: float) -> float:
        z = lin_interp(K, T)
        if np.isnan(z):
            z = nn_interp(K, T)
        return float(z)

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
