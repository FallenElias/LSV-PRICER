import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

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


def bs_implied_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    tol: float    = 1e-6,
    maxiter: int  = 100,
) -> float:
    """
    Solve for σ so that Black–Scholes call price = market_price,
    using Brent’s method on an adaptive bracket.
    """
    def objective(sigma):
        return bs_call_price(S0, K, T, r, q, sigma) - market_price

    # initial bracket [σ_low, σ_high]
    sigma_low, sigma_high = 1e-6, 5.0

    # check ends
    try:
        f_low  = objective(sigma_low)
        f_high = objective(sigma_high)
    except Exception:
        return np.nan

    # if they have same sign, expand upper bound
    if f_low * f_high > 0:
        sigma_high = 10.0
        try:
            f_high = objective(sigma_high)
        except Exception:
            return np.nan
        if f_low * f_high > 0:
            # still no sign change, give up
            return np.nan

    # now we have f_low<0<f_high (or vice-versa), so root lies in between
    try:
        iv = brentq(objective, sigma_low, sigma_high,
                    xtol=tol, maxiter=maxiter)
    except ValueError:
        return np.nan

    return iv