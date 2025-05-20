# Local Stochastic Volatility (LSV) Option Pricer GUI

This application provides a full-featured interface for pricing options on american equity using a Local Stochastic Volatility (LSV) model calibrated from real market data. It supports simulation and pricing of European, Barrier, and Asian options, with visualization of smile/skew, local volatility, and Greeks.

## Authors : Elias Dève, Arnaud Ferrand

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
git clone https://github.com/FallenElias/LSV-PRICER.git
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
│   ├── __init__.py          
│   ├── local_vol.py          # Dupire local volatility
│   ├── mc_engine.py          # Monte-Carlo simulator
│   ├── heston_calib.py       # Heston model calibration
│   ├── leverage.py           # Build leverage function L(S,t)
│   └── surface.py            # IV surface interpolation
|
├── pricing/
│   ├── __init__.py
│   ├── diagnostics.py        # Calculating RMSE between Model IV and Market IV
│   └── pricing.py            # Monte Carlo pricers (European, Barrier, Asian)
|
├── data/
│   ├── __init__.py
│   └── loader.py             # Data fetching and cleaning functions
|
├── utils/
│   ├── __init__.py
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

## User Guide

This section describes how to use the LSV Pricer application from launch to full option pricing.

### 1. Launching the Application

Ensure your virtual environment is activated, then run:

```bash
python -m gui.main
```

The application window will open with the following panels:

* **Control Panel** (left): input parameters and controls
* **Simulation and Diagnostics Panel** (right): displays plots
* **Summary Panel** (center): key metrics (spot, volatility, Greeks)
* **History Table** (bottom): recent pricing requests

---

### 2. Step-by-Step Usage

#### A. Fetch Market Data

* Enter an **american ticker symbol** (e.g., `AAPL`) Verify that market aren't close if you don't want to work offline mode
* Set **risk-free rate** `r` and **dividend yield** `q`
* (Optional) Tick `Offline Mode` if no internet; ensure `spot.csv` and `options.csv` are in the project folder
* Click **"Fetch & Clean"**
* Spot history and implied volatility surface will be loaded and plotted

#### B. Calibrate the Model

* Click **"Calibrate & Build L"**
* The app will:

  * Fit the implied volatility surface
  * Compute the Dupire local volatility surface
  * Calibrate the Heston model
  * Build the leverage function
* ATM values and summary metrics will be shown in the center panel

#### C. Configure Product Parameters

* Choose **maturity** (in years), **strike**, **option type** (Call or Put)
* Select **product type**:

  * **European**
  * **Barrier** (specify barrier level and direction: Up-and-Out, Down-and-Out, etc.)
  * **Asian** (choose Arithmetic or Geometric)
* Additional fields will appear dynamically based on product type

#### D. Simulate Paths

* Click **"Run Simulation"**
* The system will:

  * Simulate 100,000 LSV paths
  * Display 1000 sample paths in the Simulation panel
  * Compute and display spot, ATM implied/local vol, and Black-Scholes Greeks

#### E. Price the Option

* Click **"Price"**
* The app will compute the Monte Carlo price using the configured parameters
* Result will be shown in the status bar and appended to the history table

---

### 3. Viewing Diagnostics

* Diagnostics (volatility fit error) are computed automatically after calibration
* The Diagnostics tab visualizes the RMSE between fitted IV surface and market data

---

### 4. Exporting & Offline Mode

To fetch data for **offline usage**:

```python
spot.to_csv("spot.csv", index=False)
opts.to_csv("options.csv", index=False)
```

Then, enable `Offline Mode` in the app. The system will load from CSV instead of querying online APIs.


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
