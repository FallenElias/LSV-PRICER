import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import matplotlib.pyplot as plt
from utils.financial import bs_implied_vol
from model.heston_calib import heston_price


# backend modules
from data.loader           import fetch_spot_history, fetch_option_quotes, clean_option_quotes
from model.surface         import fit_iv_surface
from model.local_vol       import dupire_local_vol
from model.heston_calib    import calibrate_heston
from model.leverage        import simulate_heston, estimate_conditional_variance, build_leverage_function
from model.mc_engine import simulate_lsv_parallel as simulate_lsv
from pricing.pricing       import european_price_mc, barrier_price_mc, asian_price_mc

class LSVPricerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSV Pricer")
        self._build_controls()
        self._build_plot_area()
        
    def _build_controls(self):
        frm = ttk.Frame(self)
        frm.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker entry
        ttk.Label(frm, text="Ticker").grid(row=0, column=0, sticky="w")
        self.ticker = ttk.Entry(frm)
        self.ticker.grid(row=0, column=1)

        # Derivative type
        ttk.Label(frm, text="Product").grid(row=1, column=0, sticky="w")
        self.prod_cb = ttk.Combobox(frm, values=["European", "Barrier", "Asian"])
        self.prod_cb.current(0)
        self.prod_cb.grid(row=1, column=1)

        # Strike
        ttk.Label(frm, text="Strike").grid(row=2, column=0, sticky="w")
        self.strike = ttk.Entry(frm)
        self.strike.grid(row=2, column=1)

        # Barrier (only if Barrier)
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

        # New Diagnostics button
        self.diag_btn = ttk.Button(frm, text="Diagnostics", command=self._on_diagnostics)
        self.diag_btn.grid(row=8, column=0, columnspan=2, pady=5)

        # Result display (moved to row 9)
        self.result_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.result_var, foreground="blue") \
            .grid(row=9, column=0, columnspan=2, pady=10)


    def _build_plot_area(self):
        fig = Figure(figsize=(6,4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _on_fetch(self):
        tk = self.ticker.get().upper()
        try:
            # 1) load spot history + grab S₀
            self.spot = fetch_spot_history(tk, years=3)
            S0        = self.spot["Close"].iloc[-1]

            # 2) load option quotes with our IV inversion
            raw_opts = fetch_option_quotes(
                tk,
                S0,
                r=0.0,
                q=0.0
            )
            self.opts = clean_option_quotes(raw_opts)

            messagebox.showinfo(
                "Data",
                f"Loaded {len(self.spot)} spot rows and {len(self.opts)} option quotes"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))


            
    def _on_calibrate(self):
        """
        Kick off background calibration + leverage build with a live timer.
        """
        # disable UI controls
        self.fetch_btn.config(state="disabled")
        self.calib_btn.config(state="disabled")

        # prepare inputs
        spot_date   = self.spot["date"].iloc[-1]
        S0          = self.spot["Close"].iloc[-1]
        exp_days    = (self.opts["expiry"] - spot_date).dt.days
        times_full  = np.unique(exp_days / 365.0)
        strikes_full= np.sort(self.opts["strike"].unique())

        # down-sample to 5 × 7 grid for speed
        nT    = min(5, len(times_full))
        times = times_full[np.linspace(0, len(times_full)-1, nT, dtype=int)]
        nK      = min(7, len(strikes_full))
        strikes = strikes_full[np.linspace(0, len(strikes_full)-1, nK, dtype=int)]

        # persist the grid for diagnostics
        self.strikes = strikes
        self.times   = times

        iv_mat = np.zeros((len(times), len(strikes)))
        for i, T in enumerate(times):
            mask = np.isclose(exp_days / 365.0, T)
            slice_df = self.opts[mask]
            for j, K in enumerate(strikes):
                iv_mat[i, j] = slice_df.loc[slice_df["strike"] == K, "mid_iv"].mean()

        print("raw iv_mat:\n", iv_mat)

        # store for background task
        self._calib_inputs = (strikes, times, iv_mat, S0)
        self._last_market_iv = iv_mat.copy()


        # start timer display
        self._calib_start = time.perf_counter()
        self.result_var.set("Calibrating… 0s elapsed")
        self._timer_id = self.after(1000, self._update_timer)

        # launch background thread
        thread = threading.Thread(target=self._calibrate_task, daemon=True)
        thread.start()
        
    def _update_timer(self):
        """Update elapsed-time every second."""
        elapsed = int(time.perf_counter() - self._calib_start)
        self.result_var.set(f"Calibrating… {elapsed}s elapsed")
        self._timer_id = self.after(1000, self._update_timer)

    def _calibrate_task(self):
        """
        Runs in background. Does calibration, simulate-Heston, leverage build.
        On completion, schedules _calibrate_done on the main thread.
        """
        strikes, times, iv_mat, S0 = self._calib_inputs
        try:
            # 1) fit IV surface & local vol
            self.iv_func   = fit_iv_surface(strikes, times, iv_mat)
            self.sigma_loc = dupire_local_vol(S0, strikes, times, self.iv_func, r=0.0, q=0.0)

            # 2) calibrate Heston
            calib = calibrate_heston(strikes, times, iv_mat, S0, r=0.0, q=0.0)

            # 3) simulate pure-Heston & estimate cond. var.
            S_h, v_h = simulate_heston(
                S0, calib["v0"],
                calib["kappa"], calib["theta"], calib["xi"], calib["rho"],
                r=0.0, q=0.0,
                maturities=times,
                n_steps=len(times)-1,
                n_paths=10000
            )
            cond_var = estimate_conditional_variance(S_h, v_h, strikes, T_index=-1, bandwidth=1.0)

            # 4) build leverage
            cond_grid = np.tile(cond_var, (len(times), 1))
            Lf = build_leverage_function(strikes, times, self.sigma_loc, cond_grid)

            # signal completion on main thread
            self.after(0, lambda: self._calibrate_done(calib, Lf))

        except Exception as e:
            self.after(0, lambda err=e: messagebox.showerror("Calibration Error", str(err)))

    def _calibrate_done(self, calib, Lf):
        """
        Called on main thread when calibration + leverage build finishes.
        """
        # stop timer
        self.after_cancel(self._timer_id)
        self.result_var.set("Calibration complete")

        # store results
        self.heston_params = calib
        self.L_func         = Lf

        # re-enable buttons
        self.fetch_btn.config(state="normal")
        self.calib_btn.config(state="normal")

    
    def _on_simulate(self):
        """
        Run the LSV Monte-Carlo using self.L_func and self.heston_params.
        Plot the first 20 simulated paths.
        """
        if not hasattr(self, "L_func") or not hasattr(self, "heston_params"):
            messagebox.showwarning("Error", "Please click ‘Calibrate & Build L’ first")
            return

        S0 = self.spot["Close"].iloc[-1]
        params = self.heston_params
        T_final = self.times[-1]

        # simulate LSV
        self.S_paths, self.v_paths = simulate_lsv(
            S0,
            params["v0"],
            self.L_func,
            params,
            r=0.0,
            q=0.0,
            T=T_final,
            n_steps=100,
            n_paths=5000
        )

        # plot sample paths
        self.ax.clear()
        t_grid = np.linspace(0, T_final, 100 + 1)
        for path in self.S_paths[:20]:
            self.ax.plot(t_grid, path, linewidth=0.6)
        self.canvas.draw()

    def _on_price(self):
        """
        Price the selected derivative on self.S_paths.
        """
        if not hasattr(self, "S_paths"):
            messagebox.showwarning("Error", "Please run ‘Run Simulation’ first")
            return

        prod  = self.prod_cb.get()
        K     = float(self.strike.get())
        T     = self.times[-1]
        discount = np.exp(-0.0 * T)

        if prod == "European":
            price = european_price_mc(lambda ST: np.maximum(ST - K, 0.0),
                                     self.S_paths, discount)
        elif prod == "Barrier":
            B = float(self.barrier.get())
            price = barrier_price_mc(self.S_paths, K, B,
                                     is_up=True, is_call=True,
                                     discount=discount)
        else:  # Asian
            price = asian_price_mc(self.S_paths, K,
                                   is_call=True, discount=discount)

        self.result_var.set(f"Price: {price:.4f}")
        
    def _on_diagnostics(self):
        """
        Plot market vs model IV heatmap and print RMSE,
        re-using the last calibration without re-fitting.
        """
        # 1) Grab stored data & calib
        strikes   = self.strikes
        times     = self.times
        market_iv = self._last_market_iv
        calib     = self.heston_params
        S0        = self.spot["Close"].iloc[-1]
        r, q      = 0.0, 0.0
        
        # 2) Compute model IV surface
        model_iv = np.zeros_like(market_iv)
        for i, T in enumerate(times):
            for j, K in enumerate(strikes):
                price        = heston_price(K, T, S0, r, q, calib)
                model_iv[i,j]= bs_implied_vol(S0, K, T, r, q, price)

        # 3) Show heatmap
        error = model_iv - market_iv
        # mask off missing
        valid = ~np.isnan(error)
        if not np.any(valid):
            messagebox.showerror("Diagnostics","no valid IV points to compare")
            return
        plt.figure(figsize=(5,4))
        plt.pcolormesh(
        strikes, times, np.where(valid, error, np.nan),
        cmap="bwr", vmin=-0.02, vmax=0.02
        )
        plt.colorbar(label="Model−Market IV")
        plt.xlabel("Strike"); plt.ylabel("Maturity")
        plt.title("IV Fit Error")
        plt.show()

        # 4) Print RMSE in the GUI
        rmse = np.sqrt(np.mean(error**2))
        messagebox.showinfo("Diagnostics", f"IV fit RMSE: {rmse:.4%}")

if __name__ == "__main__":
    # 1) JIT‐warmup: compile heston_price once
    from model.heston_calib import heston_price
    heston_price(
        100.0, 1.0, 100.0,    # dummy K, T, S0
        0.01, 0.0,            # dummy r, q
        {'kappa':1.0, 'theta':0.04, 'xi':0.5, 'rho':-0.5, 'v0':0.04}
    )

    # 2) Launch the GUI
    app = LSVPricerGUI()
    app.mainloop()
