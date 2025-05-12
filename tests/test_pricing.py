# tests/test_pricing.py

import numpy as np
import pytest
from pricing.pricing import european_price_mc, barrier_price_mc, asian_price_mc


def test_european_price_mc_basic_call():
    # two paths ending at 110 and 90, strike=100, discount=1
    S_paths = np.array([[0, 110], [0, 90]])
    payoff = lambda ST: np.maximum(ST - 100, 0.0)
    price = european_price_mc(payoff, S_paths, discount=1.0)
    assert price == pytest.approx((10 + 0) / 2)


def test_barrier_price_mc_up_and_out_call():
    # two paths: one breaches up barrier, one not
    S_paths = np.array([[100, 110], [100, 102]])
    strike, barrier = 100, 105
    price = barrier_price_mc(
        S_paths, strike, barrier, is_up=True, is_call=True, discount=1.0
    )
    # first path knocked out → 0, second path payoff 2
    assert price == pytest.approx(2 / 2)


def test_asian_price_mc_basic_put():
    # two paths with averages 105 and 95, strike=100, discount=1
    S_paths = np.array([
        [100, 110, 105],  # average  (110+105)/2 = 107.5
        [100,  95,  90],  # average   (95+90)/2  = 92.5
    ])
    price_call = asian_price_mc(S_paths, strike=100, is_call=True, discount=1.0)
    price_put  = asian_price_mc(S_paths, strike=100, is_call=False, discount=1.0)

    # call payoffs: [7.5, 0] → avg = 3.75
    assert price_call == pytest.approx(3.75)
    # put payoffs: [0, 7.5] → avg = 3.75
    assert price_put == pytest.approx(3.75)
