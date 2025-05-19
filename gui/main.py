# gui/main.py

import tkinter as tk
from tkinter import ttk
from gui.frames.controls import ControlPanel
from gui.frames.summary  import SummaryPanel
from gui.frames.plots    import PlotPanel
import gui.handlers      as handlers

class LSVPrApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSV Pricer")

        # Configure layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=3)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Left: Control panel
        self.ctrl = ControlPanel(
            master=self,
            on_fetch=    lambda: handlers.start_fetch(self),
            on_calibrate=lambda: handlers.start_calib(self),
            on_simulate=lambda: handlers.start_sim(self),
            on_price=    lambda: handlers.do_price(self),
            on_diag=     lambda: handlers.run_diagnostics(self),
        )
        self.ctrl.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        # Center: summary panel
        self.summary = SummaryPanel(master=self)
        self.summary.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        # Right: Plot panel
        self.plots = PlotPanel(master=self)
        self.plots.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        # Bottom: History table
        history_frame = ttk.Frame(self)
        history_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        cols = ("Product", "Type", "K", "T", "BarrierDir", "AsianStyle", "Price")
        self.history = ttk.Treeview(
            history_frame,
            columns=cols,
            show="headings",
            height=5
        )
        for c in cols:
            self.history.heading(c, text=c)
            self.history.column(c, width=80, anchor="center")
        self.history.pack(side="left", fill="x", expand=True)

        sb = ttk.Scrollbar(history_frame, orient="vertical", command=self.history.yview)
        sb.pack(side="right", fill="y")
        self.history.configure(yscrollcommand=sb.set)


if __name__ == "__main__":
    app = LSVPrApp()
    app.mainloop()
