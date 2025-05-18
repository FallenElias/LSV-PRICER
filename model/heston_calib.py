# model/heston_calib.py
import numpy as np
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import minimize
from typing import Dict
from utils.financial import bs_implied_vol

# suppress benign warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)

# 1) Heston characteristic function

def heston_charfunc(phi: complex, S0: float, v0: float,
                   kappa: float, theta: float,
                   sigma: float, rho: float,
                   lambd: float, tau: float,
                   r: float) -> complex:
    if phi == 0:
        return 1+0j
    a = kappa * theta
    b = kappa + lambd
    iu = 1j * phi
    d = np.sqrt((rho * sigma * iu - b)**2 + (phi*1j + phi*phi) * sigma*sigma)
    g = (b - rho*sigma*iu + d) / (b - rho*sigma*iu - d)
    exp1 = np.exp(r * iu * tau)
    term2 = S0**iu * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / (sigma*sigma))
    exp2 = np.exp(
        a * tau * (b - rho*sigma*iu + d) / (sigma*sigma)
        + v0 * (b - rho*sigma*iu + d)
          * ((1 - np.exp(d*tau)) / (1 - g * np.exp(d*tau)))
          / (sigma*sigma)
    )
    return exp1 * term2 * exp2

# 2) Price via quad integration

def heston_price(K: float, T: float, S0: float,
                 r: float, q: float,
                 params: Dict[str, float]) -> float:
    v0    = params['v0'];    kappa = params['kappa']
    theta = params['theta']; sigma = params['xi']
    rho   = params['rho'];   lambd = params.get('lambd', 0.0)
    tau   = T
    def integrand(phi):
        num = np.exp(r*tau) * heston_charfunc(phi-1j, S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        num -= K * heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        den = 1j * phi * (K ** (1j*phi))
        return (num/den).real
    # avoid phi=0 singularity
    integral, _ = quad(integrand, 1e-6, 100.0, epsabs=1e-3, epsrel=1e-3, limit=50)
    forward = (S0 * np.exp(-q*T) - K * np.exp(-r*T)) / 2.0
    return max(forward + integral/np.pi, 0.0)

# 3) Helper to invert price→IV

def _model_iv_point(args):
    K, T, S0, r, q, params = args
    price = heston_price(K, T, S0, r, q, params)
    return bs_implied_vol(S0, K, T, r, q, price)

# 4) Calibration routine

def calibrate_heston(strikes: np.ndarray, maturities: np.ndarray,
                      market_iv: np.ndarray,
                      S0: float, r: float, q: float,
                      initial_guess: Dict[str, float]=None) -> Dict[str, float]:
    if initial_guess is None:
        initial_guess = {'kappa':1.0,'theta':0.04,'xi':0.5,'rho':-0.5,'v0':0.04}
    names = list(initial_guess.keys())
    x0 = np.array([initial_guess[n] for n in names])
    bounds = [(1e-3,10),(1e-4,2),(1e-4,5),(-0.999,0.999),(1e-4,2)]
    Ks, Ts = np.meshgrid(strikes, maturities)
    target = market_iv.ravel()
    def objective(x):
        params = dict(zip(names, x))
        iv_mod = [ _model_iv_point((K,T,S0,r,q,params))
                   for K,T in zip(Ks.ravel(), Ts.ravel()) ]
        return float(np.sum((np.array(iv_mod)-target)**2))
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    return dict(zip(names, res.x))


# gui/gui.py modifications:
# In _calibrate_task: down-sample strikes×times to 7×5 as shown above;
# store self.strikes_ds, self.times_ds for diagnostics.
# In _on_diagnostics: use strikes_ds/times_ds and corresponding slice of _last_market_iv
# to compute RMSE on the calibrated grid.
