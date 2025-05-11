import numpy as np
import pytest
from model.heston_calib import calibrate_heston, heston_price
from utils.financial import bs_implied_vol

def test_heston_calibration_synthetic():
    # pick “true” parameters
    true = {"kappa":2.0, "theta":0.04, "xi":0.3, "rho":-0.7, "v0":0.04}
    S0, r, q = 100.0, 0.01, 0.0

    # synthetic market: constant-vol surface => Heston with xi->0, rho irrelevant
    strikes = np.array([90,100,110])
    maturities = np.array([0.5,1.0,1.5])
    market_iv = np.full((len(maturities),len(strikes)),0.20)

    # calibrate
    calib = calibrate_heston(strikes, maturities, market_iv, S0, r, q)

    # we expect theta≈v0≈0.20², xi small
    assert pytest.approx(calib["theta"], rel=0.5) == true["theta"]
    assert pytest.approx(calib["v0"],    rel=0.5) == true["v0"]
    assert calib["xi"] < 1.0
