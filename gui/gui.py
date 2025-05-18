import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import matplotlib.pyplot as plt

from data.loader import fetch_spot_history, fetch_option_quotes, clean_option_quotes
from pricing.pricing import european_price_mc, barrier_price_mc, asian_price_mc
from model.surface import fit_iv_surface
from model.local_vol import dupire_local_vol
from model.heston_calib import calibrate_heston
from model.leverage import simulate_heston, estimate_conditional_variance, build_leverage_function
from utils.financial import bs_call_price

class LSVPricerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSV Pricer")
        self._build_controls()
        self._build_plot_area()

    def _build_controls(self):
        frm = ttk.Frame(self)
        frm.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(frm, text="Ticker").grid(row=0, column=0, sticky="w")
        self.ticker = ttk.Entry(frm)
        self.ticker.grid(row=0, column=1)

        ttk.Label(frm, text="Product").grid(row=1, column=0, sticky="w")
        self.prod_cb = ttk.Combobox(frm, values=["European","Barrier","Asian"])
        self.prod_cb.current(0)
        self.prod_cb.grid(row=1, column=1)

        ttk.Label(frm, text="Strike").grid(row=2, column=0, sticky="w")
        self.strike = ttk.Entry(frm)
        self.strike.grid(row=2, column=1)

        ttk.Label(frm, text="Barrier").grid(row=3, column=0, sticky="w")
        self.barrier = ttk.Entry(frm)
        self.barrier.grid(row=3, column=1)

        # Buttons
        self.fetch_btn = ttk.Button(frm, text="Fetch & Clean Data", command=self._on_fetch)
        self.fetch_btn.grid(row=4, column=0, columnspan=2, pady=5)

        self.calib_btn = ttk.Button(frm, text="Calibrate & Build L", command=self._on_calibrate)
        self.calib_btn.grid(row=5, column=0, columnspan=2, pady=5)

        self.sim_btn = ttk.Button(frm, text="Run Simulation", command=self._on_simulate)
        self.sim_btn.grid(row=6, column=0, columnspan=2, pady=5)

        self.price_btn = ttk.Button(frm, text="Price", command=self._on_price)
        self.price_btn.grid(row=7, column=0, columnspan=2, pady=5)

        self.diag_btn = ttk.Button(frm, text="Diagnostics", command=self._on_diagnostics)
        self.diag_btn.grid(row=8, column=0, columnspan=2, pady=5)
        self.diag_btn.config(state="disabled")

        self.result_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.result_var, foreground="blue").grid(row=9, column=0, columnspan=2, pady=10)

    def _build_plot_area(self):
        fig = Figure(figsize=(6,4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _on_fetch(self):
        ticker = self.ticker.get().upper()
        try:
            self.spot = fetch_spot_history(ticker, years=3)
            S0 = self.spot['Close'].iloc[-1]
            raw_opts = fetch_option_quotes(ticker, S0, 0.0, 0.0)
            self.opts = clean_option_quotes(raw_opts)
            messagebox.showinfo("Data", f"Loaded {len(self.spot)} spot rows and {len(self.opts)} option quotes")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_calibrate(self):
        self.fetch_btn.config(state="disabled")
        self.calib_btn.config(state="disabled")
        spot_date = self.spot['date'].iloc[-1]
        S0 = float(self.spot['Close'].iloc[-1])
        exp_days = (self.opts['expiry'] - spot_date).dt.days

        times_full = np.unique(exp_days/365.0)
        strikes_full = np.sort(self.opts['strike'].unique())
        iv_mat_full = np.full((len(times_full), len(strikes_full)), np.nan)
        for i, t in enumerate(times_full):
            iv_mat_full[i] = (
                self.opts.loc[np.isclose(exp_days/365.0, t)]
                .groupby('strike')['mid_iv'].mean()
                .reindex(strikes_full).values
            )

        M, N = len(strikes_full), len(times_full)
        nK, nT = min(7, M), min(5, N)
        idx_K = np.linspace(0, M-1, nK, dtype=int)
        idx_T = np.linspace(0, N-1, nT, dtype=int)
        strikes_ds = strikes_full[idx_K]
        times_ds = times_full[idx_T]
        iv_ds = iv_mat_full[np.ix_(idx_T, idx_K)]

        self._calib_inputs = (strikes_ds, times_ds, iv_ds, S0, strikes_full, times_full, iv_mat_full)
        self._calib_start = time.perf_counter()
        self.result_var.set("Calibrating… 0s elapsed")
        self._timer_id = self.after(1000, self._update_timer)
        threading.Thread(target=self._calibrate_task, daemon=True).start()

    def _calibrate_task(self):
        strikes_ds, times_ds, iv_ds, S0, _, _, _ = self._calib_inputs
        self.iv_func = fit_iv_surface(strikes_ds, times_ds, iv_ds)
        self.sigma_loc = dupire_local_vol(S0, strikes_ds, times_ds, self.iv_func)
        calib = calibrate_heston(strikes_ds, times_ds, iv_ds, S0, 0.0, 0.0)
        S_h, v_h = simulate_heston(
            S0, calib['v0'], calib['kappa'], calib['theta'], calib['xi'], calib['rho'],
            0.0, 0.0, times_ds, len(times_ds)-1, 10000
        )
        cond = estimate_conditional_variance(S_h, v_h, strikes_ds, T_index=-1, bandwidth=1.0)
        Lf = build_leverage_function(strikes_ds, times_ds, self.sigma_loc, np.tile(cond,(len(times_ds),1)))
        self.after_cancel(self._timer_id)
        self.after(0, lambda: self._calibrate_done(calib, Lf))

    def _calibrate_done(self, calib, Lf):
        self.heston_params, self.L_func = calib, Lf
        self.result_var.set("Calibration complete")
        self.fetch_btn.config(state="normal")
        self.calib_btn.config(state="normal")
        self.diag_btn.config(state="normal")
        messagebox.showinfo("Calibrate", "Done")

    def _on_simulate(self):
        if not hasattr(self,'L_func'):
            messagebox.showwarning("Error", "Calibrate first")
            return
        S0 = self.spot['Close'].iloc[-1]
        params = self.heston_params
        Ts = self._calib_inputs[1]
        self.S_paths, _ = simulate_heston(
            S0, params['v0'], params['kappa'], params['theta'], params['xi'], params['rho'],
            0.0, 0.0, Ts, 100, 5000
        )
        self.ax.clear()
        t = np.linspace(0, Ts[-1], 101)
        for p in self.S_paths[:20]:
            self.ax.plot(t, p, linewidth=0.6)
        self.canvas.draw()

    def _on_price(self):
        if not hasattr(self,'S_paths'):
            messagebox.showwarning("Error", "Simulate first")
            return
        prod = self.prod_cb.get()
        K = float(self.strike.get())
        T = self._calib_inputs[1][-1]
        disc = np.exp(-0.0 * T)
        if prod == "European":
            price = european_price_mc(lambda ST: np.maximum(ST - K, 0), self.S_paths, disc)
        elif prod == "Barrier":
            B = float(self.barrier.get())
            price = barrier_price_mc(self.S_paths, K, B, True, True, disc)
        else:
            price = asian_price_mc(self.S_paths, K, True, disc)
        self.result_var.set(f"Price: {price:.4f}")

    def _on_diagnostics(self):
        if not hasattr(self, 'iv_func'):
            messagebox.showerror("Diagnostics", "Please click 'Calibrate & Build L' first.")
            return

        # Unpack
        strikes_ds, times_ds, iv_ds, S0, allK, allT, iv_full = self._calib_inputs

        # Slice full IV
        idxK = [int(np.where(allK == k)[0][0]) for k in strikes_ds]
        idxT = [int(np.where(allT == t)[0][0]) for t in times_ds]
        m_iv = iv_full[np.ix_(idxT, idxK)]

        # Compute RMSE over all valid points
        ivf = self.iv_func
        model_all  = np.array([ivf(k, t) for t in times_ds for k in strikes_ds])
        market_all = m_iv.ravel()
        mask_all   = np.isfinite(model_all) & np.isfinite(market_all)
        if not np.any(mask_all):
            messagebox.showerror("Diagnostics", "No valid IV points.")
            return
        errs_all = model_all[mask_all] - market_all[mask_all]
        rmse = np.sqrt((errs_all**2).mean())

        # Rebuild valid (i,j) pairs for plotting
        ii_full, jj_full = np.where(~np.isnan(m_iv))
        valid = [
            (i, j) for (i, j), keep in zip(zip(ii_full, jj_full), mask_all)
            if keep
        ]
        ii = np.array([i for i, _ in valid])
        jj = np.array([j for _, j in valid])

        # Compute errors for each plotted point
        errs_plot = np.array([
            ivf(strikes_ds[j], times_ds[i]) - m_iv[i, j]
            for i, j in valid
        ])

        # Plot
        self.ax.clear()
        self.ax.set_xlabel("Strike")
        self.ax.set_ylabel("Maturity")
        self.ax.set_title(f"IV Fit Error (RMSE {rmse:.2%})")
        # 2) Auto‐scale axes to the new data
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.scatter(
            strikes_ds[jj],
            times_ds[ii],
            c=errs_plot,
            cmap='bwr',
            vmin=-0.02,
            vmax=0.02,
            edgecolors='k',
            linewidths=0.7,
            s=60
        )
        self.canvas.draw()
        messagebox.showinfo("Diagnostics", f"IV fit RMSE: {rmse:.2%}")

    def _update_timer(self):
        elapsed = int(time.perf_counter() - self._calib_start)
        self.result_var.set(f"Calibrating… {elapsed}s elapsed")
        self._timer_id = self.after(1000, self._update_timer)

if __name__ == "__main__":
    app = LSVPricerGUI()
    app.mainloop()
