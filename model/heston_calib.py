import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from typing import Dict
from utils.financial import bs_implied_vol  
from numba import njit, complex128, float64
from scipy.integrate import quad, IntegrationWarning
import time

from multiprocessing.dummy import Pool as ThreadPool


_CALIB_POOL = ThreadPool()


import warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)

import inspect

print("Loading heston_calib from:", inspect.getsourcefile(inspect.getmodule(inspect.currentframe())))



# Gauss–Laguerre nodes & weights for ∫₀^∞ e^{-u} g(u) du
_GL_N = 16
_GL_x, _GL_w = np.polynomial.laguerre.laggauss(_GL_N)


# 1) Characteristic function accepts complex128 u
@njit(complex128(complex128,  # u (real or shifted complex)
                 float64,    # lnS0
                 float64,    # T
                 float64,    # r
                 float64,    # q
                 float64,    # kappa
                 float64,    # theta
                 float64,    # xi
                 float64,    # rho
                 float64))   # v0
def _heston_cf_numba(u, lnS0, T, r, q, kappa, theta, xi, rho, v0):
    iu = 1j * u
    d  = np.sqrt((rho*xi*iu - kappa)**2 + xi*xi * (u*u + iu))
    g  = (kappa - rho*xi*iu - d) / (kappa - rho*xi*iu + d)
    exp_dT = np.exp(-d * T)
    C = (r - q)*iu*T + (theta*kappa/(xi*xi)) * (
            (kappa - rho*xi*iu - d)*T
            - 2.0 * np.log((1.0 - g*exp_dT)/(1.0 - g))
        )
    D = ((kappa - rho*xi*iu - d)/(xi*xi)) * ((1.0 - exp_dT)/(1.0 - g*exp_dT))
    return np.exp(C + D*v0 + iu * lnS0)

# 2) Single integrand for both P1 and P2
@njit(float64(complex128,   # u
              float64,   # lnK
              float64,   # lnS0
              float64,   # T
              float64,   # r
              float64,   # q
              float64,   # kappa
              float64,   # theta
              float64,   # xi
              float64,   # rho
              float64))  # v0
def _integrand_numba(u, lnK, lnS0, T, r, q, kappa, theta, xi, rho, v0):
    # computing for j=1 or j=2 is handled by shifting u externally
    cf = _heston_cf_numba(u, lnS0, T, r, q, kappa, theta, xi, rho, v0)
    num = np.exp(-1j * u * lnK) * cf
    return (num / (1j * u)).real

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
    Price a European call option under the Heston stochastic‐volatility model
    using Fourier inversion and fixed‐node Gauss–Laguerre quadrature.

    Formula:
      C(K,T) = S0 * exp(-q T) * P1 − K * exp(-r T) * P2
    where for j=1,2:
      Pj = ½ + (1/π) ∫₀^∞ Re[ e^{-i u ln K} φ(u - i(j-1)) / (i u) ] du

    We approximate the infinite integral ∫₀^∞ f(u) du by a 16‐point
    Gauss–Laguerre rule:
      ∫₀^∞ f(u) du ≈ Σ_{i=1..16} w_i * exp(x_i) * f(x_i).

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    S0 : float
        Current spot price.
    r : float
        Risk‐free interest rate (annualized, continuous).
    q : float
        Dividend yield (annualized, continuous).
    params : dict
        Heston parameters:
          'kappa' : mean‐reversion speed of variance (κ)
          'theta' : long‐run variance (θ)
          'xi'    : volatility of volatility (ξ)
          'rho'   : correlation between asset and variance (ρ)
          'v0'    : initial variance at t=0

    Returns
    -------
    price : float
        The model price of a European call.
    """

    
    # Coerce inputs to native Python floats for consistency
    K  = float(np.array(K).item())
    T  = float(np.array(T).item())
    S0 = float(np.array(S0).item())
    r  = float(r)
    q  = float(q)

    kappa, theta, xi, rho, v0 = map(float, (
        params["kappa"],
        params["theta"],
        params["xi"],
        params["rho"],
        params["v0"],
    ))

    def P(j: int) -> float:
        """
        Compute P1 or P2 via fixed‐node Gauss–Laguerre quadrature.
        The shift j-1 moves the argument of the characteristic function.
        """
        lnK  = np.log(K)         # log-strike
        lnS0 = np.log(S0)        # log-spot
        shift = np.complex128(1j * (j - 1))

        def f(u: float) -> float:
            """
            Integrand f(u) = Re[ e^{-i u ln K} * φ(u - i(j-1)) / (i u) ]
            evaluated at u = real Gauss–Laguerre node.
            """
            u_c = np.complex128(u) - shift
            # φ = characteristic function from numba‐compiled routine
            cf = _heston_cf_numba(
                u_c, lnS0, T, r, q,
                kappa, theta, xi, rho, v0
            )
            # multiply by the Fourier kernel and divide by i u, take real part
            numer = np.exp(-1j * u * lnK) * cf
            return float((numer / (1j * u)).real)

        # evaluate f at all GL nodes, weight and sum
        values   = np.array([f(xi) for xi in _GL_x])
        weighted = _GL_w * np.exp(_GL_x)
        integral = np.dot(weighted, values)

        # return Pj = ½ + (1/π) * integral
        return 0.5 + integral / np.pi

    # compute the two probabilities
    P1 = P(1)
    P2 = P(2)
    # assemble the Black‐Scholes‐style price
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
        # 1st call: record t0
        if not hasattr(objective, "calls"):
            objective.calls = 0
            objective.t0   = time.perf_counter()
        elif objective.calls == 1:
            print("First objective took",
                time.perf_counter() - objective.t0, "s")

        # increment & report
        objective.calls += 1
        print(f">>> objective call #{objective.calls}")

        params   = dict(zip(param_names, x))
        args_list = [(K, T, S0, r, q, params)
                    for K, T in zip(Ks_flat, Ts_flat)]

        # fan out work to the pre-started worker processes
        iv_mod = _CALIB_POOL.map(_model_iv_point, args_list)

        errs = (np.array(iv_mod) - target_iv) ** 2
        return float(errs.sum())


    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    return dict(zip(param_names, result.x))
