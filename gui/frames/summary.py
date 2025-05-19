# gui/frames/summary.py

import tkinter as tk
from tkinter import ttk

class SummaryPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        ttk.Label(self, text="Spot:").grid(row=0, column=0, sticky="w")
        self.lbl_spot = ttk.Label(self, text="—")
        self.lbl_spot.grid(row=0, column=1, sticky="e")

        ttk.Label(self, text="ATM Implied Vol:").grid(row=1, column=0, sticky="w")
        self.lbl_iv = ttk.Label(self, text="—")
        self.lbl_iv.grid(row=1, column=1, sticky="e")

        ttk.Label(self, text="ATM Local Vol:").grid(row=2, column=0, sticky="w")
        self.lbl_loc = ttk.Label(self, text="—")
        self.lbl_loc.grid(row=2, column=1, sticky="e")

        # Header for greeks
        ttk.Label(
            self,
            text="BS Greeks (ATM European)",
            font=("TkDefaultFont", 9, "italic"),
            foreground="gray"
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Greeks
        greeks = ["delta", "gamma", "vega", "theta", "rho"]
        for i, g in enumerate(greeks, start=4):
            ttk.Label(self, text=g.capitalize() + ":").grid(row=i, column=0, sticky="w")
            setattr(self, f"lbl_{g}", ttk.Label(self, text="—"))
            getattr(self, f"lbl_{g}").grid(row=i, column=1, sticky="e")

