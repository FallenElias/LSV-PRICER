import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.integrate import quad
from scipy.optimize import minimize
from typing import Dict
from utils.financial import bs_implied_vol  # your BS‐IV solver


def _heston_cf(
    u: complex,
    S0: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
) -> complex:
    """
    Compute the Heston model characteristic function φ(u; T).

    φ(u) = E[e^{i u ln S_T}] under risk-neutral dynamics:
      dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW^S_t
      dv_t = κ(θ - v_t) dt + ξ sqrt(v_t) dW^v_t,  Corr(dW^S, dW^v)=ρ dt

    Parameters
    ----------
    u      : complex
        Argument of the characteristic function.
    S0     : float
        Initial spot price S(0).
    T      : float
        Time to maturity (years).
    r      : float
        Continuous risk-free rate.
    q      : float
        Continuous dividend yield.
    kappa  : float
        Mean-reversion speed of variance.
    theta  : float
        Long-run mean of variance.
    xi     : float
        Volatility of variance (“vol-of-vol”).
    rho    : float
        Correlation between asset and variance Brownian motions.
    v0     : float
        Initial variance at time zero.

    Returns
    -------
    φ : complex
        The characteristic function value at u.
    """
    a = kappa * theta
    iu = 1j * u
    # d and g are intermediate terms in the closed-form solution
    d = np.sqrt((rho*xi*iu - kappa)**2 + xi**2*(iu + u**2))
    g = (kappa - rho*xi*iu - d) / (kappa - rho*xi*iu + d)
    exp_dT = np.exp(-d * T)

    # C(T,u) term in exponent
    C = (r * iu * T
         + (a / xi**2)
           * ((kappa - rho*xi*iu - d) * T
              - 2.0 * np.log((1 - g * exp_dT) / (1 - g))))
    # D(T,u) term multiplying v0 in exponent
    D = ((kappa - rho*xi*iu - d) / xi**2) * ((1 - exp_dT) / (1 - g * exp_dT))

    # shift spot by dividends: use S0 e^{-qT}
    return np.exp(C + D * v0 + iu * np.log(S0 * np.exp(-q * T)))


def heston_price(
    K: float,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: Dict[str, float],
) -> float:
    """
    Price a European call under Heston via Fourier inversion.

    C(K,T) = S0 e^{-qT} P1 − K e^{-rT} P2

    where
      Pj = ½ + (1/π) ∫₀^∞ Re[ e^{−i u ln K} φ(u − i(j−1)) / (i u) ] du

    Parameters
    ----------
    K      : float
        Strike price.
    T      : float
        Time to maturity (years).
    S0     : float
        Spot price.
    r      : float
        Risk-free rate.
    q      : float
        Dividend yield.
    params : dict
        Heston parameters:
          'kappa' : mean-reversion speed (κ)
          'theta' : long-run variance (θ)
          'xi'    : vol-of-vol (ξ)
          'rho'   : correlation (ρ)
          'v0'    : initial variance

    Returns
    -------
    price : float
        Heston model European call price.
    """
    # Unpack parameters
    kappa, theta, xi, rho, v0 = (
        params["kappa"], params["theta"], params["xi"],
        params["rho"], params["v0"]
    )

    def P(j: int) -> float:
        def integrand(u: float) -> float:
            # 1) get the raw characteristic‐function value
            raw = _heston_cf(u - 1j*(j-1), S0, T, r, q,
                            kappa, theta, xi, rho, v0)
            # 2) if it’s a pandas Series of length 1, extract its element
            if isinstance(raw, pd.Series):
                raw = raw.iloc[0]
            # 3) now force to a built‐in complex
            cf_val = complex(raw)

            num = np.exp(-1j * u * np.log(K)) * cf_val
            real_part = (num / (1j * u)).real
            return float(real_part)

        integral, _ = quad(integrand, 0.0, np.inf, limit=200)
        return 0.5 + (1.0 / np.pi) * integral

    P1 = P(1)
    P2 = P(2)
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

def _model_iv_point(args):
    K, T, S0, r, q, params = args
    price = heston_price(K, T, S0, r, q, params)
    return bs_implied_vol(S0, K, T, r, q, price)

def calibrate_heston(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_iv: np.ndarray,
    S0: float,
    r: float,
    q: float,
    initial_guess: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Find Heston parameters that best match a grid of market implied vols.

    We minimize:
      ∑_{i,j} [ σ_model(Kj,T_i) − σ_market(Kj,T_i) ]^2

    where σ_model is obtained by:
      1. compute Heston call price C_model(K,T)
      2. invert Black–Scholes to implied vol

    Parameters
    ----------
    strikes      : 1D array
        Strike grid Kj.
    maturities   : 1D array
        Maturity grid T_i.
    market_iv    : 2D array
        Market implied vols σ_market[i,j] at each (T_i, K_j).
    S0           : float
        Spot price.
    r            : float
        Risk-free rate.
    q            : float
        Dividend yield.
    initial_guess: dict, optional
        Starting values for parameters:
          {
            'kappa': ...,
            'theta': ...,
            'xi': ...,
            'rho': ...,
            'v0': ...
          }

    Returns
    -------
    calibrated : dict
        Optimized parameters with keys 'kappa','theta','xi','rho','v0'.
    """
    # Build flat arrays for K, T and target IV
    Ks, Ts = np.meshgrid(strikes, maturities)
    Ks_flat = Ks.ravel()
    Ts_flat = Ts.ravel()
    target_iv = market_iv.ravel()

    if initial_guess is None:
        initial_guess = {
            'kappa': 1.0,
            'theta': 0.04,
            'xi': 0.5,
            'rho': -0.5,
            'v0': 0.04,
        }
    x0 = np.array(list(initial_guess.values()))
    bounds = [
        (1e-3, 10.0),    # kappa
        (1e-4, 2.0),     # theta
        (1e-4, 5.0),     # xi
        (-0.999, 0.999), # rho
        (1e-4, 2.0),     # v0
    ]
    param_names = list(initial_guess.keys())

    def objective(x):
        params = dict(zip(param_names, x))
        # prepare args list once per objective call
        args_list = [
            (K, T, S0, r, q, params)
            for K, T in zip(Ks_flat, Ts_flat)
        ]
        # parallel map
        with mp.Pool(mp.cpu_count()) as pool:
            iv_mod = pool.map(_model_iv_point, args_list)
        errs = (np.array(iv_mod) - target_iv) ** 2
        return float(errs.sum())

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    return dict(zip(param_names, result.x))
