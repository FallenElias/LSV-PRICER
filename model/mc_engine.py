
import numpy as np
from typing import Callable, Dict, Tuple


def simulate_lsv(
    S0: float,
    v0: float,
    leverage_func: Callable[[float, float], float],
    heston_params: Dict[str, float],
    r: float,
    q: float,
    T: float,
    n_steps: int,
    n_paths: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Local Stochastic Volatility paths under risk-neutral dynamics:

      dS_t = (r - q) S_t dt + L(S_t, t) sqrt(v_t) S_t dW^S_t
      dv_t = κ(θ - v_t) dt + ξ sqrt(v_t) dW^v_t,
      Corr(dW^S, dW^v) = ρ

    Uses Euler‐Maruyama with full‐truncation on v_t.

    Parameters
    ----------
    S0           : float
        Initial spot.
    v0           : float
        Initial variance.
    leverage_func: callable
        L(S, t) leverage function.
    heston_params: dict
        {"kappa","theta","xi","rho"} for variance dynamics.
    r            : float
        Risk‐free rate.
    q            : float
        Dividend yield.
    T            : float
        Time to maturity (years).
    n_steps      : int
        Number of time steps.
    n_paths      : int
        Number of Monte‐Carlo paths.

    Returns
    -------
    (S_paths, v_paths) : tuple of arrays
        S_paths, v_paths have shape (n_paths, n_steps+1).
    """
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi    = heston_params["xi"]
    rho   = heston_params["rho"]

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full((n_paths, n_steps+1), S0, dtype=float)
    v = np.full((n_paths, n_steps+1), v0, dtype=float)

    for t in range(n_steps):
        # generate correlated normals
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_paths)

        v_prev = v[:, t]
        # Heston full‐truncation Euler step for variance
        v_next = np.maximum(
            v_prev
            + kappa * (theta - np.maximum(v_prev, 0.0)) * dt
            + xi * np.sqrt(np.maximum(v_prev, 0.0)) * sqrt_dt * z2,
            0.0,
        )

        # LSV spot step
        t_now = t * dt
        L = leverage_func(S[:, t], t_now)  # vectorize if needed
        S[:, t + 1] = S[:, t] * np.exp(
            (r - q - 0.5 * v_prev * L**2) * dt + L * np.sqrt(v_prev * dt) * z1
        )

        v[:, t + 1] = v_next

    return S, v
