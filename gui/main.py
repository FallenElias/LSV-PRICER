# gui/main.py

import tkinter as tk
from gui.frames.controls import ControlPanel
from gui.frames.plots    import PlotPanel
import gui.handlers      as handlers

class LSVPrApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSV Pricer")

        self.ctrl = ControlPanel(
            master=self,
            on_fetch=    lambda: handlers.start_fetch(self),
            on_calibrate=lambda: handlers.start_calib(self),
            on_simulate=lambda: handlers.start_sim(self),
            on_price=    lambda: handlers.do_price(self),
            on_diag=     lambda: handlers.run_diagnostics(self),
        )
        self.ctrl.pack(side="left", fill="y", padx=10, pady=10)

        self.plots = PlotPanel(master=self)
        self.plots.pack(side="right", fill="both", expand=True, padx=10, pady=10)

if __name__ == "__main__":
    app = LSVPrApp()
    app.mainloop()
