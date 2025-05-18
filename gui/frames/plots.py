# gui/frames/plots.py

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# gui/frames/plots.py

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotPanel(ttk.Notebook):
    def __init__(self, master):
        super().__init__(master)

        # History tab
        self.history_tab = ttk.Frame(self)
        self.add(self.history_tab, text="History")
        self.canvas_hist = self._make_canvas(self.history_tab)

        # Smile/Skew tab
        self.smile_tab = ttk.Frame(self)
        self.add(self.smile_tab, text="Smile/Skew")
        self.canvas_smile = self._make_canvas(self.smile_tab)

        # Simulation paths tab
        self.sim_tab = ttk.Frame(self)
        self.add(self.sim_tab, text="Simulation")
        self.canvas_sim = self._make_canvas(self.sim_tab)

        # Diagnostics tab
        self.diag_tab = ttk.Frame(self)
        self.add(self.diag_tab, text="Diagnostics")
        self.canvas_diag = self._make_canvas(self.diag_tab)

    def _make_canvas(self, frame):
        fig = Figure(figsize=(6, 4))
        fig.add_subplot(111)                  # create axes
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # add nav toolbar anchored under the canvas
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()

        return canvas

