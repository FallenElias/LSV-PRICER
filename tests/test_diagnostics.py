# tests/test_diagnostics.py

import numpy as np
import pytest
from pricing.diagnostics import check_mc_convergence
from model.mc_engine import simulate_lsv
from pricing.pricing    import european_price_mc
from model.heston_calib import calibrate_heston

def test_check_mc_convergence_european_call():
    # set up a trivial LSV engine with L=1 so price matches BS
    S0, v0, r, q = 100.0, 0.04, 0.01, 0.0
    strikes    = np.array([100.0])
    maturities = np.array([1.0])
    market_iv  = np.array([[0.20]])
    calib = calibrate_heston(strikes, maturities, market_iv, S0, r, q)
    iv_func = lambda K, T: 0.20

    # build leverage = 1
    from model.leverage import simulate_heston, estimate_conditional_variance, build_leverage_function
    times = maturities
    S_h, v_h = simulate_heston(S0, calib["v0"],
                                calib["kappa"], calib["theta"], calib["xi"], calib["rho"],
                                r, q,
                                times,
                                n_steps=1, n_paths=5000)
    cond_var = estimate_conditional_variance(S_h, v_h, strikes, T_index=-1, bandwidth=1.0)
    L_func = build_leverage_function(strikes, maturities,
                                     np.array([[0.20]]),
                                     np.array([[cond_var[0]]]))

    # define price_func wrapper
    def price_func(n_steps, n_paths):
        S_paths, _ = simulate_lsv(S0, calib["v0"], L_func, calib, r, q,
                                  T=maturities[0], n_steps=n_steps, n_paths=n_paths)
        return european_price_mc(lambda ST: np.maximum(ST - 100, 0.0),
                                 S_paths,
                                 np.exp(-r*maturities[0]))

    step_list = [10, 50]
    path_list = [1000, 5000]
    errors = check_mc_convergence(price_func,
                                  args={},  # price_func wraps all else
                                  step_list=step_list,
                                  path_list=path_list)
    # errors should decrease as n_steps and n_paths increase
    assert errors[1,1] < errors[0,1]  # finer steps better
    assert errors[1,1] < errors[1,0]  # more paths better
