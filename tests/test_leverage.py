import numpy as np
import pytest
from model.leverage import simulate_heston, estimate_conditional_variance, build_leverage_function

def test_simulate_heston_constant_variance():
    # if xi=0, kappa arbitrary, v0 fixed, variance is constant v0
    S0, v0 = 100.0, 0.04
    kappa, theta, xi, rho = 1.0, 0.04, 0.0, 0.0
    r, q = 0.0, 0.0
    maturities = np.array([0.0, 1.0])
    S_paths, v_paths = simulate_heston(
        S0, v0, kappa, theta, xi, rho, r, q, maturities, n_steps=10, n_paths=5000
    )
    # at final time all v_t should equal v0
    assert np.allclose(v_paths[:, -1], v0, atol=1e-3)

def test_estimate_conditional_variance_uniform():
    # create artificial S_T and v_T where v_T = 0.04 + 0.01*(S_T - 100)
    ST = np.linspace(90, 110, 1000)
    vT = 0.04 + 0.01*(ST - 100)
    S_paths = np.vstack([ST, ST]).T  # shape (1000,2)
    v_paths = np.vstack([vT, vT]).T
    K_grid = np.array([95, 100, 105])
    cond = estimate_conditional_variance(S_paths, v_paths, K_grid, T_index=1, bandwidth=0.5)
    # at each K_grid[i], cond[i] ≈ 0.04 + 0.01*(K_grid[i]-100)
    expected = 0.04 + 0.01*(K_grid - 100)
    assert np.allclose(cond, expected, atol=1e-3)

def test_build_leverage_constant():
    """
    If σ_loc and cond_var are both constant, L = σ_loc / sqrt(cond_var) is constant.
    """
    sigma_loc = 0.2
    cond_var   = 0.04  # variance = (0.2)^2
    # build small grid
    strikes    = np.array([90.0, 100.0, 110.0])
    maturities = np.array([0.5, 1.0, 1.5])
    lv_grid = np.full((len(maturities), len(strikes)), sigma_loc)
    cv_grid = np.full_like(lv_grid, cond_var)

    L = build_leverage_function(strikes, maturities, lv_grid, cv_grid)

    # query at grid and off-grid points
    for K in [92.0, 100.0, 108.0]:
        for T in [0.6, 1.2]:
            assert pytest.approx(L(K, T), rel=1e-6) == sigma_loc / np.sqrt(cond_var)