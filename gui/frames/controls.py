# gui/frames/controls.py

import tkinter as tk
from tkinter import ttk

class ControlPanel(ttk.Frame):
    def __init__(self, master,
                 on_fetch, on_calibrate, on_simulate, on_price, on_diag):
        super().__init__(master)
        self.status = tk.StringVar(value="Ready")

        # Row 0: Ticker
        ttk.Label(self, text="Ticker").grid(row=0, column=0, sticky="w")
        self.ticker = ttk.Entry(self)
        self.ticker.grid(row=0, column=1)

        # Row 1: Risk-free rate r
        ttk.Label(self, text="r").grid(row=1, column=0, sticky="w")
        self.r = ttk.Entry(self); self.r.insert(0, "0.0")
        self.r.grid(row=1, column=1)

        # Row 2: Dividend yield q
        ttk.Label(self, text="q").grid(row=2, column=0, sticky="w")
        self.q = ttk.Entry(self); self.q.insert(0, "0.0")
        self.q.grid(row=2, column=1)

        # Row 3: Maturity (years)
        ttk.Label(self, text="Maturity").grid(row=3, column=0, sticky="w")
        self.maturity = ttk.Entry(self)
        self.maturity.grid(row=3, column=1)

        # Row 4: Strike
        ttk.Label(self, text="Strike").grid(row=4, column=0, sticky="w")
        self.strike = ttk.Entry(self)
        self.strike.grid(row=4, column=1)

        # Row 5: Option Type
        ttk.Label(self, text="Opt Type").grid(row=5, column=0, sticky="w")
        self.opt_type = ttk.Combobox(self, values=["Call","Put"], state="readonly")
        self.opt_type.current(0)
        self.opt_type.grid(row=5, column=1)

        # Row 6: Product
        ttk.Label(self, text="Product").grid(row=6, column=0, sticky="w")
        self.product = ttk.Combobox(
            self, values=["European","Barrier","Asian"], state="readonly"
        )
        self.product.current(0)
        self.product.grid(row=6, column=1)

        # Barrier / Asian controls (will be shown/hidden)
        self.barrier_label = ttk.Label(self, text="Barrier")
        self.barrier = ttk.Entry(self)
        self.barrier_dir_label = ttk.Label(self, text="Barrier Dir")
        self.barrier_dir = ttk.Combobox(
            self,
            values=["Up-and-Out","Up-and-In","Down-and-Out","Down-and-In"],
            state="readonly"
        )
        self.barrier_dir.current(0)

        self.asian_label = ttk.Label(self, text="Asian Style")
        self.asian = ttk.Combobox(
            self, values=["Arithmetic","Geometric"], state="readonly"
        )
        self.asian.current(0)

        self.product.bind("<<ComboboxSelected>>", self._on_product_change)


        # ── Buttons (rows 17–20) ─────────────────────────────────────────────
        self.fetch_btn = ttk.Button(self, text="Fetch & Clean",   command=on_fetch)
        row = 20  # or any number after your last used row index
        self.fetch_btn.grid(row=row, column=0, columnspan=2, pady=2); row += 1
        self.calib_btn = ttk.Button(self, text="Calibrate & Build L", command=on_calibrate)
        self.calib_btn.grid(row=row, column=0, columnspan=2, pady=2); row += 1
        self.sim_btn   = ttk.Button(self, text="Run Simulation", command=on_simulate)
        self.sim_btn.grid(row=row, column=0, columnspan=2, pady=2); row += 1
        self.price_btn = ttk.Button(self, text="Price",           command=on_price)
        self.price_btn.grid(row=row, column=0, columnspan=2, pady=2); row += 1

        # ── Status (row 21) ───────────────────────────────────────────────────
        ttk.Label(self, textvariable=self.status, foreground="blue")\
            .grid(row=row, column=0, columnspan=2, pady=5)

        # initial hide/show
        self._on_product_change()
        
        # Add at the end of __init__ in gui/frames/controls.py
        self.use_offline = tk.BooleanVar()
        ttk.Checkbutton(self, text="Offline mode (CSV)", variable=self.use_offline).grid(row=1+row, column=0, columnspan=2, sticky="w")



    def _on_product_change(self, event=None):
        # hide all
        for w in (self.barrier_label, self.barrier,
                  self.barrier_dir_label, self.barrier_dir,
                  self.asian_label,   self.asian):
            w.grid_forget()

        # show only what we need
        p = self.product.get()
        if p == "Barrier":
            self.barrier_label.grid(row=7, column=0, sticky="w")
            self.barrier.grid(row=7, column=1)
            self.barrier_dir_label.grid(row=7, column=0, sticky="w")
            self.barrier_dir.grid(row=8, column=1)
        elif p == "Asian":
            self.asian_label.grid(row=7, column=0, sticky="w")
            self.asian.grid(row=7, column=1)
