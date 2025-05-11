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
    tol: float = 1e-6,
    maxiter: int = 100,
) -> float:
    """
    Solve for σ such that Black–Scholes call price equals market_price.

    Parameters
    ----------
    S0           : float
        Spot price.
    K            : float
        Strike.
    T            : float
        Time to maturity.
    r            : float
        Risk-free rate.
    q            : float
        Dividend yield.
    market_price : float
        Observed call price.
    tol          : float
        Solver tolerance.
    maxiter      : int
        Maximum iterations for root finding.

    Returns
    -------
    implied_vol : float
        The volatility σ that matches market_price.
    """
    # Define objective: BS_price(sigma) - market_price = 0
    def objective(sigma):
        return bs_call_price(S0, K, T, r, q, sigma) - market_price

    # Vol bounds: [1e-6, 5.0]
    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=maxiter)
    except ValueError:
        # if price outside achievable range, return NaN
        return np.nan
    return iv
