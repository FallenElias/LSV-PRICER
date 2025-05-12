import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# backend modules
from data.loader           import fetch_spot_history, fetch_option_quotes, clean_option_quotes
from model.surface         import fit_iv_surface
from model.local_vol       import dupire_local_vol
from model.heston_calib    import calibrate_heston
from model.leverage        import simulate_heston, estimate_conditional_variance, build_leverage_function
from model.mc_engine       import simulate_lsv
from pricing.pricing       import european_price_mc, barrier_price_mc, asian_price_mc

class LSVPricerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSV Pricer")
        self._build_controls()
        self._build_plot_area()

    def _build_controls(self):
        frm = ttk.Frame(self); frm.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker entry
        ttk.Label(frm, text="Ticker").grid(row=0, column=0, sticky="w")
        self.ticker = ttk.Entry(frm); self.ticker.grid(row=0, column=1)

        # Derivative type
        ttk.Label(frm, text="Product").grid(row=1, column=0, sticky="w")
        self.prod_cb = ttk.Combobox(frm, values=["European","Barrier","Asian"])
        self.prod_cb.current(0); self.prod_cb.grid(row=1, column=1)

        # Strike
        ttk.Label(frm, text="Strike").grid(row=2, column=0, sticky="w")
        self.strike = ttk.Entry(frm); self.strike.grid(row=2, column=1)

        # Barrier (only if Barrier)
        ttk.Label(frm, text="Barrier").grid(row=3, column=0, sticky="w")
        self.barrier = ttk.Entry(frm); self.barrier.grid(row=3, column=1)

        # Buttons
        ttk.Button(frm, text="Fetch & Clean Data", command=self._on_fetch).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(frm, text="Calibrate & Build L", command=self._on_calibrate).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(frm, text="Run Simulation", command=self._on_simulate).grid(row=6, column=0, columnspan=2, pady=5)
        ttk.Button(frm, text="Price", command=self._on_price).grid(row=7, column=0, columnspan=2, pady=5)

        # Result display
        self.result_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.result_var, foreground="blue").grid(row=8, column=0, columnspan=2, pady=10)

    def _build_plot_area(self):
        fig = Figure(figsize=(6,4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _on_fetch(self):
        tk = self.ticker.get().upper()
        try:
            self.spot = fetch_spot_history(tk)
            opts = fetch_option_quotes(tk)
            self.opts = clean_option_quotes(opts)
            messagebox.showinfo("Data", f"Loaded {len(self.spot)} spot rows and {len(self.opts)} options")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_calibrate(self):
        """
        Build IV surface, local vol, calibrate Heston, and build leverage function.
        """
        # spot price and date
        spot_date = self.spot["date"].iloc[-1]
        S0 = self.spot["Close"].iloc[-1]

        # compute time-to-expiry in years, unique & sorted
        exp_days = (self.opts["expiry"] - spot_date).dt.days
        self.times = np.unique(exp_days / 365.0)

        # strike grid
        strikes = np.sort(self.opts["strike"].unique())

        # build mid-IV matrix on (len(times), len(strikes))
        iv_mat = np.zeros((len(self.times), len(strikes)))
        for i, T in enumerate(self.times):
            mask = np.isclose(exp_days / 365.0, T)
            slice_df = self.opts[mask]
            for j, K in enumerate(strikes):
                iv_mat[i, j] = slice_df.loc[slice_df["strike"] == K, "mid_iv"].mean()

        # 1) fit implied-vol surface
        self.iv_func = fit_iv_surface(strikes, self.times, iv_mat)

        # 2) extract local volatility
        self.sigma_loc = dupire_local_vol(S0, strikes, self.times, self.iv_func, r=0.0, q=0.0)

        # 3) calibrate Heston backbone
        calib = calibrate_heston(strikes, self.times, iv_mat, S0, r=0.0, q=0.0)
        self.heston_params = calib

        # 4) simulate pure-Heston for conditional variance
        S_h, v_h = simulate_heston(
            S0,
            calib["v0"],
            calib["kappa"], calib["theta"], calib["xi"], calib["rho"],
            r=0.0, q=0.0,
            maturities=self.times,
            n_steps=len(self.times) - 1,
            n_paths=10000
        )
        cond_var = estimate_conditional_variance(S_h, v_h, strikes, T_index=-1, bandwidth=1.0)

        # 5) build leverage L(K,T)
        cond_grid = np.tile(cond_var, (len(self.times), 1))
        self.L_func = build_leverage_function(strikes, self.times, self.sigma_loc, cond_grid)

        messagebox.showinfo("Calibrate", "Calibration and leverage build complete")

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

if __name__ == "__main__":
    app = LSVPricerGUI()
    app.mainloop()
