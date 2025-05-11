import numpy as np
import pytest
from model.surface import fit_iv_surface, svi_parametrize


def test_fit_iv_surface_linear():
    # simple test: iv = 0.1 + 0.02*K + 0.05*T
    Ks = np.array([90, 100, 110])
    Ts = np.array([0.5, 1.0, 1.5])
    IV = np.empty((len(Ts), len(Ks)))
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            IV[i, j] = 0.1 + 0.02*K + 0.05*T

    iv_func = fit_iv_surface(Ks, Ts, IV, kx=1, ky=1)  # linear spline
    # test at grid points
    for T in Ts:
        for K in Ks:
            assert pytest.approx(iv_func(K, T), rel=1e-6) == 0.1 + 0.02*K + 0.05*T
    # test at off-grid point
    K0, T0 = 95, 1.25
    expected = 0.1 + 0.02*K0 + 0.05*T0
    assert pytest.approx(iv_func(K0, T0), rel=1e-3) == expected


def test_svi_parametrize_not_implemented():
    # ensure stub raises NotImplementedError
    with pytest.raises(NotImplementedError):
        svi_parametrize(np.array([0.0]), np.array([0.1]))
