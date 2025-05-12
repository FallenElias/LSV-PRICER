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
        # build IV grid
        strikes = np.sort(self.opts["strike"].unique())
        exps    = np.sort(self.opts["expiry"].dt.year + self.opts["expiry"].dt.dayofyear/365)
        # form mid_iv matrix...
        iv_mat  = np.zeros((len(exps), len(strikes)))
        for i,T in enumerate(exps):
            row = self.opts[self.opts["expiry"].dt.year + self.opts["expiry"].dt.dayofyear/365 == T]
            iv_mat[i,:] = [row[row["strike"]==K]["mid_iv"].mean() for K in strikes]
        self.iv_func = fit_iv_surface(strikes, exps, iv_mat)
        # local vol
        S0 = self.spot["Close"].iloc[-1]
        self.sigma_loc = dupire_local_vol(S0, strikes, exps, self.iv_func)
        # Heston calib
        calib = calibrate_heston(strikes, exps, iv_mat, S0, r=0.0, q=0.0)
        # leverage
        times = exps
        S_h, v_h = simulate_heston(S0, calib["v0"], **calib, r=0.0, q=0.0, maturities=times, n_steps=len(times)-1, n_paths=10000)
        cond_var = estimate_conditional_variance(S_h, v_h, strikes, T_index=-1)
        self.L_func = build_leverage_function(strikes, exps, self.sigma_loc, np.tile(cond_var,(len(exps),1)))
        messagebox.showinfo("Calibrate", "Heston and leverage built")

    def _on_simulate(self):
        S0 = self.spot["Close"].iloc[-1]
        calib = self.L_func  # not needed
        self.S_paths, _ = simulate_lsv(S0, calib["v0"], self.L_func, calib, r=0.0, q=0.0, T=1.0, n_steps=100, n_paths=5000)
        self.ax.clear()
        for i in range(20):
            self.ax.plot(self.S_paths[i], linewidth=0.6)
        self.canvas.draw()

    def _on_price(self):
        prod = self.prod_cb.get()
        K = float(self.strike.get())
        disc = np.exp(-0.0 * 1.0)
        if prod == "European":
            p = european_price_mc(lambda ST: np.maximum(ST-K,0), self.S_paths, disc)
        elif prod == "Barrier":
            B = float(self.barrier.get())
            p = barrier_price_mc(self.S_paths, K, B, is_up=True, is_call=True, discount=disc)
        else:
            p = asian_price_mc(self.S_paths, K, is_call=True, discount=disc)
        self.result_var.set(f"Price: {p:.4f}")
