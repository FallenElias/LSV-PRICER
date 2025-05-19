# Local Stochastic Volatility (LSV) Option Pricer GUI

This application provides a full-featured interface for pricing american equity options using a Local Stochastic Volatility (LSV) model calibrated from real market data. It supports simulation and pricing of European, Barrier, and Asian options, with visualization of smile/skew, local volatility, and Greeks.

---

## 🧠 Key Features

* **LSV Model Implementation**

  * Combines Dupire's local volatility with Heston stochastic volatility.
  * Calibration on implied volatility surface from market option quotes.

* **Option Products Supported**

  * European (Call/Put)
  * Barrier (Up-and-Out, Down-and-Out, Up-and-In, Down-and-In)
  * Asian (Arithmetic and Geometric)

* **Pricing**

  * Monte Carlo simulations on calibrated LSV paths.
  * Pricing engine for each product.
  * Summary panel with Greeks and implied/local volatility.

* **GUI**

  * Built in `Tkinter` with real-time interaction.
  * Historical spot and IV visualization.
  * Diagnostics, log, and price history panel.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lsv-option-pricer.git
cd lsv-option-pricer
```

### 2. Create and Activate Virtual Environment

```bash
conda create -n lsv_env python=3.10 -y
conda activate lsv_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
python -m gui.main
```

---

## 📂 Project Structure

```
lsv-option-pricer/
|
├── gui/
│   ├── frames/
│   │   ├── controls.py       # Sidebar controls and user input
│   │   ├── plots.py          # Plotting panels (history, smile, sim, diagnostics)
│   │   ├── summary.py        # Greeks and volatility panel
│   │   └── __init__.py
│   ├── gui.py                # [Deprecated] Old interface
│   ├── handlers.py           # All event logic (fetch, calibrate, simulate, price)
│   └── main.py               # Main application launcher
|
├── model/
│   ├── local_vol.py          # Dupire local volatility
│   ├── heston_calib.py       # Heston model calibration
│   ├── leverage.py           # Build leverage function L(S,t)
│   └── surface.py            # IV surface interpolation
|
├── pricing/
│   └── pricing.py            # Monte Carlo pricers (European, Barrier, Asian)
|
├── data/
│   └── loader.py             # Data fetching and cleaning functions
|
├── utils/
│   └── financial.py          # BS formulas, implied vol, greeks, helpers
|
├── spot.csv                  # [Optional] Offline spot price backup
├── options.csv               # [Optional] Offline option quotes backup
├── requirements.txt
└── README.md
```

---

## 📦 Offline Mode

To work offline:

1. Tick the **Offline Mode** checkbox in the GUI.
2. Ensure the following CSVs are in the root directory:

* `spot.csv`: Must contain columns `date` (datetime), `Close` (float).
* `options.csv`: Must contain cleaned option data including `expiry`, `strike`, `mid_iv`.

These files are loaded automatically and bypass online API calls.

---

## 🧪 Functionality Flow

1. **Fetch Data**

   * Download historical spot prices and market option quotes.
   * Clean and preprocess to build IV grid.
   * Supports offline fallback using local CSVs.

2. **Calibration**

   * Fit implied volatility surface.
   * Compute local volatility via Dupire’s PDE.
   * Calibrate Heston parameters.
   * Compute leverage function L(S,t) from conditional variance.

3. **Simulation**

   * Run Monte Carlo simulations of the LSV model.
   * Plot sample spot paths.

4. **Pricing**

   * Compute price using simulated paths.
   * Display summary: spot, IV, local vol, Greeks (BS approximation).
   * Store results in log and history table.

5. **Visualization**

   * Smile/skew plot from market IV surface.
   * Spot price history.
   * IV diagnostics (RMSE scatter).
   * Payoff shape preview (coming soon).

---

## 📉 Greeks Displayed

The Greeks are computed using the Black-Scholes model at the ATM point with the calibrated implied volatility and used for indicative purposes:

* **Delta**: Sensitivity to spot changes.
* **Gamma**: Sensitivity of delta to spot.
* **Vega**: Sensitivity to volatility.
* **Theta**: Time decay.
* **Rho**: Sensitivity to interest rate.

---

## 🧬 Notes & Limitations

* Greeks are Black-Scholes based, not from the LSV model.
* Barrier pricer uses Knock-Out logic; Knock-In supported via parity.
* Geometric Asian options now supported.
* Pricing accuracy depends on simulation size and calibration quality.
* Use 100,000 paths for stable estimates.

---

## 📞 Support

For issues or feature requests, open an issue on the [GitHub repository](https://github.com/yourusername/lsv-option-pricer/issues).

---

## 📜 License

This project is for academic and non-commercial use only.
