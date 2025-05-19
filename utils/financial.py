import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def bs_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black–Scholes European call price.

    Parameters
    ----------
    S0    : float
        Spot price at t=0.
    K     : float
        Option strike.
    T     : float
        Time to maturity (years).
    r     : float
        Continuous risk-free rate.
    q     : float
        Continuous dividend yield.
    sigma : float
        Volatility.

    Returns
    -------
    price : float
        Call option price.
    """
    if T <= 0:
        return max(S0 - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return (S0*np.exp(-q*T)*norm.cdf(d1)
            - K*np.exp(-r*T)*norm.cdf(d2))


import numpy as np
from scipy.optimize import brentq

def bs_implied_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> float:
    """
    Solve for σ so that Black–Scholes call price = market_price,
    using Brent’s method on an adaptive bracket.
    """
    # coerce everything to native Python floats
    S0           = float(S0)
    K            = float(K)
    T            = float(T)
    r            = float(r)
    q            = float(q)
    market_price = float(market_price)

    def objective(sigma):
        # bs_call_price is already pure‐float‐based
        return bs_call_price(S0, K, T, r, q, sigma) - market_price

    # initial bracket [σ_low, σ_high]
    sigma_low, sigma_high = 1e-6, 5.0

    # evaluate at the ends
    try:
        f_low  = objective(sigma_low)
        f_high = objective(sigma_high)
    except Exception:
        return np.nan

    # if no sign change, try to expand
    if f_low * f_high > 0:
        sigma_high = 10.0
        try:
            f_high = objective(sigma_high)
        except Exception:
            return np.nan
        if f_low * f_high > 0:
            return np.nan

    try:
        iv = brentq(objective, sigma_low, sigma_high,
                    xtol=tol, maxiter=maxiter)
    except ValueError:
        return np.nan

    return iv


def bs_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> Dict[str, float]:
    """
    Black–Scholes Greeks for European option:
      Δ, Γ, Vega, Theta, Rho
    (for a *call*; for a put, Δ and Θ and Rho adjust via put-call parity)
    """
    if T <= 0:
        return dict(delta=float(S0 > K),
                    gamma=0.0,
                    vega=0.0,
                    theta=0.0,
                    rho=0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    delta = np.exp(-q*T)*cdf_d1
    gamma = np.exp(-q*T)*pdf_d1/(S0*sigma*sqrtT)
    vega  = S0*np.exp(-q*T)*pdf_d1*sqrtT
    # Theta for call
    theta = (
        - (S0*sigma*np.exp(-q*T)*pdf_d1)/(2*sqrtT)
        - r*K*np.exp(-r*T)*cdf_d2
        + q*S0*np.exp(-q*T)*cdf_d1
    )
    # Rho for call
    rho = K*T*np.exp(-r*T)*cdf_d2

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
