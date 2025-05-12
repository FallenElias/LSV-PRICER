import numpy as np
from scipy.stats import norm
from typing import Callable
from utils.financial import bs_call_price


def dupire_local_vol(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_func: Callable[[float, float], float],
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """
    Compute local volatility σ_loc(K,T) via Dupire’s formula in the forward measure.

    Steps:
      1. Build call‐price grid C[i,j] = BS_price(S0, K_j, T_i, r, q, iv_func(K_j, T_i))
      2. Discount‐remove: C_tilde[i,j] = exp(r * T_i) * C[i,j]
      3. Approximate ∂_T C_tilde and ∂^2_KK C by central finite differences
      4. σ_loc^2 = 2 * ∂_T C_tilde / (K^2 * ∂^2_KK C)
      5. Take square root; boundaries remain NaN

    Parameters
    ----------
    S0         : float
        Current spot price.
    strikes    : 1D np.ndarray, shape (M,)
        Strike grid (sorted).
    maturities : 1D np.ndarray, shape (N,)
        Maturity grid in years (sorted).
    iv_func    : Callable[[K, T], float]
        Function returning σ_imp(K,T).
    r          : float
        Risk‐free rate.
    q          : float
        Dividend yield.

    Returns
    -------
    sigma_loc : 2D np.ndarray, shape (N, M)
        Local volatility surface; NaN at boundaries.
    """
    Ks = np.asarray(strikes, dtype=float)
    Ts = np.asarray(maturities, dtype=float)
    N, M = len(Ts), len(Ks)

    # 1. build BS call price grid
    C = np.empty((N, M))
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            vol = iv_func(K, T)
            vol = float(vol)  
            C[i, j] = bs_call_price(S0, K, T, r, q, vol)

    # 2. forward‐measure adjustment
    # C_tilde[i,j] = e^{r*T_i} * C[i,j]
    C_tilde = np.exp(r * Ts)[:, None] * C

    sigma_loc2 = np.full_like(C, np.nan)

    # 3. finite differences on interior points
    for i in range(1, N - 1):
        dt = Ts[i + 1] - Ts[i - 1]
        for j in range(1, M - 1):
            # ∂_T C_tilde
            dCtilde_dT = (C_tilde[i + 1, j] - C_tilde[i - 1, j]) / dt

            # ∂^2_KK C on original C
            dK_forward = Ks[j + 1] - Ks[j]
            dK_backward = Ks[j] - Ks[j - 1]
            d2C_dK2 = (
                (C[i, j + 1] - C[i, j]) / dK_forward
                - (C[i, j] - C[i, j - 1]) / dK_backward
            ) / ((dK_forward + dK_backward) / 2)

            denom = (Ks[j] ** 2) * d2C_dK2
            if denom > 0 and dCtilde_dT > 0:
                sigma_loc2[i, j] = 2.0 * dCtilde_dT / denom

    # 4. return sqrt of variance
    return np.sqrt(sigma_loc2)
