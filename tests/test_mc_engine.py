# tests/test_mc_engine.py

import numpy as np
import pytest
from model.mc_engine import simulate_lsv
from model.leverage import simulate_heston

def test_simulate_lsv_shapes_and_positive():
    # trivial leverage = 1 => behaves like Heston
    S0, v0 = 100.0, 0.04
    hparams = {"kappa":1.0, "theta":0.04, "xi":0.2, "rho":0.0}
    r, q = 0.01, 0.0
    T = 1.0
    n_steps, n_paths = 10, 500

    L1 = lambda S, t: 1.0
    S_lsv, v_lsv = simulate_lsv(S0, v0, L1, hparams, r, q, T, n_steps, n_paths)

    assert S_lsv.shape == (n_paths, n_steps+1)
    assert v_lsv.shape == (n_paths, n_steps+1)
    assert np.all(v_lsv >= 0)  # variance non-negative

def test_simulate_lsv_reduces_to_heston_when_L1():
    S0, v0 = 100.0, 0.04
    hparams = {"kappa":1.5, "theta":0.04, "xi":0.3, "rho":0.2}
    r, q = 0.0, 0.0
    T = 0.5
    n_steps, n_paths = 50, 2000

    # simulate pure Heston
    S_h, v_h = simulate_heston(
        S0, v0,
        hparams["kappa"], hparams["theta"], hparams["xi"], hparams["rho"],
        r, q,
        np.linspace(0, T, n_steps+1),
        n_steps, n_paths
    )

    # simulate LSV with L=1
    L1 = lambda S, t: 1.0
    S_lsv, v_lsv = simulate_lsv(S0, v0, L1, hparams, r, q, T, n_steps, n_paths)

    # spot means should match within 1%
    mean_h = S_h.mean(axis=0)
    mean_l = S_lsv.mean(axis=0)
    assert np.allclose(mean_h, mean_l, rtol=1e-2)

    # variance remains non-negative and initial mean close to v0
    assert np.all(v_lsv >= 0)
    assert pytest.approx(v_lsv[:, 0].mean(), rel=1e-6) == v0