# gui/handlers.py

import threading
import time
import numpy as np
import pandas as pd
from tkinter import messagebox
from matplotlib.dates import DateFormatter

from data.loader        import fetch_spot_history, fetch_option_quotes, clean_option_quotes
from model.surface      import fit_iv_surface
from model.local_vol    import dupire_local_vol
from model.heston_calib import calibrate_heston
from model.leverage     import simulate_heston, estimate_conditional_variance, build_leverage_function
from utils.financial    import bs_greeks

# ——— 1) Fetch & Clean Data ——————————————————————————————————————
def plot_history(app):
    df = app.spot
    canvas = app.plots.canvas_hist
    ax = canvas.figure.axes[0]
    ax.clear()
    ax.plot(df['date'], df['Close'], linewidth=1)
    ax.set_title("Spot Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    canvas.figure.autofmt_xdate()      
    canvas.draw()
    
def plot_market_smile(app):
    """
    After fetching raw data, show the market-implied-vol curve
    at the longest expiry in your down-sampled grid.
    """
    # unpack full grid you computed in start_fetch
    strikes = app.strikes_full
    times   = app.times_full
    iv_full = app.iv_full

    # pick the last expiry slice
    i = -1  
    canvas = app.plots.canvas_smile
    ax     = canvas.figure.axes[0]
    ax.clear()
    ax.plot(strikes, iv_full[i,:], marker='o', linestyle='-')
    ax.set_title(f"Market IV Smile @ T={times[i]:.3f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol")
    canvas.draw()


def start_fetch(app):
    """
    Fetch spot and option data.
    If 'Offline Mode' checkbox is ticked, load CSVs instead of fetching online.
    """

    offline = bool(app.ctrl.use_offline.get())  # read checkbox

    try:
        if offline:
            # Load offline CSVs
            app.spot = pd.read_csv("spot.csv", parse_dates=["date"])
            app.opts = pd.read_csv("options.csv", parse_dates=["expiry"])
            messagebox.showinfo("Offline Mode", f"Loaded offline data: {len(app.spot)} spot rows, {len(app.opts)} option quotes")
        else:
            # Online fetch
            from data.loader import fetch_spot_history, fetch_option_quotes, clean_option_quotes
            ticker = app.ctrl.ticker.get().upper()
            app.spot = fetch_spot_history(ticker, years=3)
            S0 = float(app.spot['Close'].iloc[-1])
            r = float(app.ctrl.r.get() or 0.0)
            q = float(app.ctrl.q.get() or 0.0)
            raw = fetch_option_quotes(ticker, S0, r, q)
            app.opts = clean_option_quotes(raw)
            messagebox.showinfo("Data", f"Loaded {len(app.spot)} spot rows and {len(app.opts)} option quotes")

        # Precompute IV surface for diagnostics and smile plotting
        spot_date = app.spot['date'].iloc[-1]
        exp_days  = (app.opts['expiry'] - spot_date).dt.days
        times_full   = np.unique(exp_days / 365.0)
        strikes_full = np.sort(app.opts['strike'].unique())
        iv_full = np.full((len(times_full), len(strikes_full)), np.nan)
        for i, T in enumerate(times_full):
            df = app.opts[np.isclose(exp_days / 365.0, T)]
            iv_full[i, :] = df.groupby('strike')['mid_iv'].mean().reindex(strikes_full).values

        app.times_full   = times_full
        app.strikes_full = strikes_full
        app.iv_full      = iv_full

        # Plot
        plot_market_smile(app)
        plot_history(app)

        app.ctrl.status.set("Fetch successful (offline)" if offline else "Fetch successful")

    except Exception as e:
        messagebox.showerror("Fetch Error", str(e))


# ——— 2) Calibrate & Build Leverage —————————————————————————————
def start_calib(app):
    """
    Down-sample IV grid to 7×5 strikes×maturities,
    fit IV surface, local-vol, Heston calib, simulate & build L.
    Runs in background thread.
    """
    # disable buttons to prevent reentry
    app.ctrl.fetch_btn.config(state="disabled")
    app.ctrl.calib_btn.config(state="disabled")

    # unpack spot & opts
    spot_date = app.spot['date'].iloc[-1]
    S0 = float(app.spot['Close'].iloc[-1])
    exp_days = (app.opts['expiry'] - spot_date).dt.days
    times_full   = np.unique(exp_days / 365.0)
    strikes_full = np.sort(app.opts['strike'].unique())

    # build full market IV matrix
    iv_full = np.full((len(times_full), len(strikes_full)), np.nan)
    for i, T in enumerate(times_full):
        slice_df = app.opts[np.isclose(exp_days/365.0, T)]
        iv_full[i,:] = slice_df.groupby('strike')['mid_iv'].mean().reindex(strikes_full).values

    # down-sample indices
    M,N    = len(strikes_full), len(times_full)
    idx_K  = np.linspace(0, M-1, min(7,M), dtype=int)
    idx_T  = np.linspace(0, N-1, min(5,N), dtype=int)
    strikes_ds = strikes_full[idx_K]
    times_ds   = times_full[idx_T]
    iv_ds      = iv_full[np.ix_(idx_T, idx_K)]

    # store for later
    app._calib_inputs = (strikes_ds, times_ds, iv_ds, S0, strikes_full, times_full, iv_full)

    # populate maturity combobox
    #app.ctrl.maturity.config(values=[f"{t:.6f}" for t in times_ds])
    #app.ctrl.maturity.current(len(times_ds)-1)

    # start timer
    app._calib_start = time.perf_counter()
    app.ctrl.status.set("Calibrating… 0s elapsed")
    app._timer_id = app.after(1000, lambda: _update_timer(app))

    # background work
    thread = threading.Thread(target=_calib_task, args=(app,), daemon=True)
    thread.start()


def _calib_task(app):
    strikes_ds, times_ds, iv_ds, S0, _, _, _ = app._calib_inputs

    # 1) IV surface
    app.ivf = fit_iv_surface(strikes_ds, times_ds, iv_ds)

    # 2) Dupire local vol
    app.sigma_loc = dupire_local_vol(S0, strikes_ds, times_ds, app.ivf)
    
    # 3) Heston calibration
    calib = calibrate_heston(strikes_ds, times_ds, iv_ds, S0, 0.0, 0.0)
    app.heston_params = calib

    # 4) Simulate pure Heston to get conditional var
    S_h, v_h = simulate_heston(
        S0, calib['v0'], calib['kappa'], calib['theta'],
        calib['xi'], calib['rho'],
        0.0, 0.0,
        times_ds, len(times_ds)-1, 10000
    )
    cond = estimate_conditional_variance(S_h, v_h, strikes_ds, T_index=-1, bandwidth=1.0)

    # 5) Leverage function
    app.Lfunc = build_leverage_function(
        strikes_ds, times_ds, app.sigma_loc,
        np.tile(cond, (len(times_ds),1))
    )

    # on complete
    app.after_cancel(app._timer_id)
    app.after(0, lambda: _calib_done(app))
    



def _calib_done(app):
    app.ctrl.status.set("Calibration complete")
    for btn in (app.ctrl.fetch_btn, app.ctrl.calib_btn):
        btn.config(state="normal")
    messagebox.showinfo("Calibration", "Done")
    run_diagnostics(app)        


# ——— 3) Simulation —————————————————————————————————————————————
def start_sim(app):
    """
    Simulate LSV paths to the maturity entered.
    Reads r, q, and maturity; updates summary labels;
    runs simulate_heston; plots on the simulation canvas.
    """
    from tkinter import messagebox
    import numpy as np
    from model.leverage import simulate_heston

    # 0) Ensure calibration done
    if not hasattr(app, 'Lfunc'):
        messagebox.showwarning("Simulate", "Please calibrate first")
        return

    # 1) Read common inputs
    S0 = float(app.spot['Close'].iloc[-1])
    p  = app.heston_params

    # 2) Read & validate r, q
    try:
        r = float(app.ctrl.r.get().strip() or 0.0)
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid r: {app.ctrl.r.get()}")
        return

    try:
        q = float(app.ctrl.q.get().strip() or 0.0)
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid q: {app.ctrl.q.get()}")
        return

    # 3) Read & validate maturity T
    Tstr = app.ctrl.maturity.get().strip()
    try:
        T = float(Tstr)
        if T <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid positive maturity (in years).")
        return

    # 4) Update summary labels now that we know T
    strikes_ds, times_ds, iv_ds, S0_calib, _, _, _ = app._calib_inputs
    atm_iv  = app.ivf(S0, T)
    iK = int(np.argmin(np.abs(strikes_ds - S0)))
    iT = int(np.argmin(np.abs(times_ds   - T)))
    atm_loc = app.sigma_loc[iT, iK]
    
    # compute BS Greeks at (S0,K=S0,T,r,q,atm_iv)
    greeks = bs_greeks(S0, S0, T, r=0.0, q=0.0, sigma=atm_iv)

    # stash for UI
    app._atm_summary = dict(
    spot=S0,
    iv=atm_iv,
    loc=atm_loc,
    **greeks
    )
    
    s = app._atm_summary
    app.summary.lbl_spot .config(text=f"{s['spot']:.2f}")
    app.summary.lbl_iv   .config(text=f"{s['iv']*100:.2f}%")
    app.summary.lbl_loc  .config(text=f"{s['loc']*100:.2f}%")
    app.summary.lbl_delta.config(text=f"{s['delta']:.4f}")
    app.summary.lbl_gamma.config(text=f"{s['gamma']:.6f}")
    app.summary.lbl_vega .config(text=f"{s['vega']:.2f}")
    app.summary.lbl_theta.config(text=f"{s['theta']:.4f}")
    app.summary.lbl_rho  .config(text=f"{s['rho']:.4f}")

    # 5) Run the LSV simulation
    n_steps = 100
    n_paths = 100_000
    S_paths, _ = simulate_heston(
        S0, p['v0'], p['kappa'], p['theta'], p['xi'], p['rho'],
        r, q,
        maturities=np.array([T]),
        n_steps=n_steps,
        n_paths=n_paths
    )
    app.S_paths = S_paths

    # 6) Plot first 500 sample paths
    canvas = app.plots.canvas_sim
    ax     = canvas.figure.axes[0]
    ax.clear()
    ax.set_title("Sample LSV Paths")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Spot price")
    t_grid = np.linspace(0, T, n_steps+1)
    for path in S_paths[:2000]:
        ax.plot(t_grid, path, linewidth=0.6)
    canvas.draw()


# ——— 4) Pricing ————————————————————————————————————————————————
def do_price(app):
    """
    Price the selected payoff on the simulated LSV paths.
    Supports European, Barrier (knock-out), and Asian (Arithmetic & Geometric) options,
    with Call/Put selection, Barrier direction, and Asian style.
    """
    from tkinter import messagebox
    import numpy as np
    import time
    from pricing.pricing import (
        european_price_mc,
        barrier_price_mc,
        asian_price_mc,
        geometric_asian_price_mc
    )

    # 0) Ensure simulation has run
    if not hasattr(app, 'S_paths'):
        messagebox.showwarning("Price", "Run simulation first")
        return

    # 1) Validate inputs
    try:
        T = float(app.ctrl.maturity.get().strip())
        if T <= 0:
            raise ValueError("Maturity must be positive")
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid maturity: {e}")
        return

    try:
        r = float(app.ctrl.r.get().strip() or 0.0)
        q = float(app.ctrl.q.get().strip() or 0.0)
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid rate: {e}")
        return

    try:
        K = float(app.ctrl.strike.get())
    except Exception:
        messagebox.showerror("Input Error", "Invalid strike")
        return

    prod     = app.ctrl.product.get()
    is_call  = (app.ctrl.opt_type.get() == "Call")
    discount = np.exp(-r * T)

    # 2) Initialize extra info
    barrier_dir = ""
    asian_style = ""

    # 3) Compute price
    if prod == "European":
        payoff = (lambda ST: np.maximum(ST - K, 0.0)) if is_call \
               else (lambda ST: np.maximum(K - ST, 0.0))
        price = european_price_mc(payoff, app.S_paths, discount)

    elif prod == "Barrier":
        try:
            B = float(app.ctrl.barrier.get())
        except Exception:
            messagebox.showerror("Input Error", "Invalid barrier level")
            return

        bd = app.ctrl.barrier_dir.get()
        is_up = "Up" in bd
        price = barrier_price_mc(
            S_paths  = app.S_paths,
            strike   = K,
            barrier  = B,
            is_up    = is_up,
            is_call  = is_call,
            discount = discount
        )
        barrier_dir = bd

    elif prod == "Asian":
        style = app.ctrl.asian.get()
        asian_style = style

        if style == "Arithmetic":
            price = asian_price_mc(
                S_paths  = app.S_paths,
                strike   = K,
                is_call  = is_call,
                discount = discount
            )
        elif style == "Geometric":
            price = geometric_asian_price_mc(
                S_paths  = app.S_paths,
                strike   = K,
                is_call  = is_call,
                discount = discount
            )
        else:
            messagebox.showerror("Input Error", f"Unknown Asian style: {style}")
            return
    else:
        messagebox.showerror("Input Error", f"Unsupported product: {prod}")
        return

    # 4) Display result
    app.ctrl.status.set(f"Price: {price:.4f}")

    # 5) Log to history (latest on top)
    app.history.insert(
        parent="",
        index=0,
        iid=None,
        values=(
            prod,
            "Call" if is_call else "Put",
            f"{K:.1f}",
            f"{T:.3f}",
            barrier_dir,
            asian_style,
            f"{price:.4f}"
        )
    )


# ——— 5) Diagnostics ——————————————————————————————————————————————
def run_diagnostics(app):
    """
    Plot IV fit error scatter on the diag_tab canvas.
    Uses app.ivf, app._calib_inputs.
    """
    if not hasattr(app, 'ivf'):
        messagebox.showerror("Diagnostics", "Calibrate first")
        return

    strikes_ds, times_ds, iv_ds, _, allK, allT, iv_full = app._calib_inputs

    # slice full IV
    idxK = [np.where(allK==k)[0][0] for k in strikes_ds]
    idxT = [np.where(allT==t)[0][0] for t in times_ds]
    m_iv = iv_full[np.ix_(idxT, idxK)]

    # compute errors
    model = np.array([app.ivf(k,t) for t in times_ds for k in strikes_ds])
    market= m_iv.ravel()
    mask  = np.isfinite(model)&np.isfinite(market)
    errs  = (model-market)[mask]
    rmse  = np.sqrt((errs**2).mean())

    ii_full, jj_full = np.where(~np.isnan(m_iv))
    valid = []
    errs_plot = []
    for (i,j) in zip(ii_full, jj_full):
        m_val = app.ivf(strikes_ds[j], times_ds[i])
        mk   = m_iv[i,j]
        if np.isfinite(m_val) and np.isfinite(mk):
            valid.append((i,j))
            errs_plot.append(m_val - mk)

    if not valid:
        messagebox.showerror("Diagnostics", "No valid IV points.")
        return

    ii = [i for i,_ in valid]
    jj = [j for _,j in valid]

    # plot
    canvas = app.plots.canvas_diag
    ax = canvas.figure.axes[0]
    ax.clear()
    ax.set_title(f"IV Fit Error (RMSE {rmse:.2%})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.relim(); ax.autoscale_view()
    ax.scatter(
        strikes_ds[jj],
        times_ds[ii],
        c=errs_plot,           # now matches x,y lengths
        cmap='bwr', vmin=-0.02, vmax=0.02,
        edgecolors='k', linewidths=0.7, s=60
    )
    sc = ax.scatter(
    strikes_ds[jj],
    times_ds[ii],
    c=errs_plot, cmap='bwr',
    vmin=-0.02, vmax=0.02,
    edgecolors='k', linewidths=0.7, s=60
    )

    # add colorbar
    canvas = app.plots.canvas_diag
    fig    = canvas.figure
    fig.colorbar(sc, ax=ax, label="Model − Market IV")
    canvas.draw()


# ——— Utility: timer update ————————————————————————————————————————
def _update_timer(app):
    elapsed = int(time.perf_counter() - app._calib_start)
    app.ctrl.status.set(f"Calibrating… {elapsed}s elapsed")
    app._timer_id = app.after(1000, lambda: _update_timer(app))



