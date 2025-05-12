import numpy as np
from typing import Callable


def european_price_mc(
    payoff_func: Callable[[np.ndarray], np.ndarray],
    S_paths: np.ndarray,
    discount: float,
) -> float:
    """
    Monte‐Carlo pricing of a European payoff.

    Parameters
    ----------
    payoff_func : callable
        Function mapping terminal asset prices array (shape (n_paths,)) to payoffs.
    S_paths     : np.ndarray, shape (n_paths, n_steps+1)
        Simulated asset‐price paths.
    discount    : float
        Discount factor = exp(-r T).

    Returns
    -------
    price : float
        Estimated present value E[payoff] * discount.
    """
    ST = S_paths[:, -1]
    payoffs = payoff_func(ST)
    return discount * np.mean(payoffs)


def barrier_price_mc(
    S_paths: np.ndarray,
    strike: float,
    barrier: float,
    is_up: bool,
    is_call: bool,
    discount: float,
) -> float:
    """
    Monte‐Carlo pricing of a single‐barrier (knock‐out) option.

    Knock‐out option: if underlying breaches barrier at any time, payoff=0.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_steps+1)
        Simulated asset‐price paths.
    strike  : float
        Option strike.
    barrier : float
        Barrier level.
    is_up    : bool
        True for up‐barrier, False for down‐barrier.
    is_call  : bool
        True for call payoff, False for put payoff.
    discount : float
        Discount factor exp(-r T).

    Returns
    -------
    price : float
        Estimated present value.
    """
    # Determine breach for each path
    if is_up:
        breached = np.any(S_paths >= barrier, axis=1)
    else:
        breached = np.any(S_paths <= barrier, axis=1)

    ST = S_paths[:, -1]
    if is_call:
        raw_payoff = np.maximum(ST - strike, 0.0)
    else:
        raw_payoff = np.maximum(strike - ST, 0.0)

    payoffs = np.where(breached, 0.0, raw_payoff)
    return discount * np.mean(payoffs)


def asian_price_mc(
    S_paths: np.ndarray,
    strike: float,
    is_call: bool,
    discount: float,
) -> float:
    """
    Monte‐Carlo pricing of an arithmetic‐average Asian option.

    Parameters
    ----------
    S_paths : np.ndarray, shape (n_paths, n_steps+1)
        Simulated asset‐price paths.
    strike  : float
        Option strike.
    is_call : bool
        True for call, False for put.
    discount : float
        Discount factor exp(-r T).

    Returns
    -------
    price : float
    """
    avg = np.mean(S_paths[:, 1:], axis=1)  # exclude S0
    if is_call:
        payoffs = np.maximum(avg - strike, 0.0)
    else:
        payoffs = np.maximum(strike - avg, 0.0)
    return discount * np.mean(payoffs)