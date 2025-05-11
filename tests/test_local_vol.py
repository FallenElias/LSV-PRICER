import numpy as np
import pytest
from model.local_vol import dupire_local_vol

def test_dupire_flat_vol_zero_rates():
    """
    A flat implied-vol surface should yield local vol ≈ constant
    when r = q = 0, up to finite-difference error.
    """
    sigma0 = 0.2
    Ks = np.linspace(80, 120, 9)
    Ts = np.linspace(0.5, 2.0, 7)
    iv_func = lambda K, T: sigma0

    S0 = 100.0
    r = 0.0
    q = 0.0

    sigma_loc = dupire_local_vol(S0, Ks, Ts, iv_func, r, q)
    inner = sigma_loc[1:-1, 1:-1]

    # Shape check
    assert inner.shape == (len(Ts) - 2, len(Ks) - 2)

    # Finite-difference error tolerance: max abs error < 0.005
    max_err = np.max(np.abs(inner - sigma0))
    assert max_err < 5e-3, f"max error {max_err:.4f} exceeds tolerance"

    # Boundaries should still be NaN
    assert np.isnan(sigma_loc[0, :]).all()
    assert np.isnan(sigma_loc[-1, :]).all()
    assert np.isnan(sigma_loc[:, 0]).all()
    assert np.isnan(sigma_loc[:, -1]).all()
