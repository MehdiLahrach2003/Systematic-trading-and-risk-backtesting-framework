# Quant Journey

**Quant Journey** is a personal research and development project in **quantitative finance**, **risk management**, and **strategy backtesting**, developed as part of my **M1 in Applied Mathematics – Statistics major (Université Paris-Dauphine)**.

The objective is to build a **coherent and reproducible quant framework**, covering the full pipeline:

> data → strategies → backtesting → risk management → portfolio allocation → derivatives pricing → results analysis.

---

## Project Objectives

* Implement a **modular and extensible backtesting engine**
* Study and compare different **quantitative trading strategies**
* Implement **risk management methodologies** (VaR, ES, risk contributions)
* Explore **multi-asset portfolio allocation**
* Implement **derivatives pricing and hedging models**
* Produce **reproducible analytical and graphical results**

This project serves both as an **advanced learning support** and a **technical portfolio** oriented toward quant / trading / research roles.

---

## Project Architecture

backtesting/ — Backtesting engine and strategies

* engine.py : main backtesting loop, time handling and position management

* rules.py : trading rules and signal generation

* portfolio.py : portfolio construction and tracking

* optimizer.py : portfolio optimization (minimum variance, efficient frontier, risk parity)

* regime.py : market regime filters

* Risk management

  * risk.py : global risk logic

  * risk_measures.py : risk measures (VaR, ES, drawdown, etc.)

  * risk_contrib.py : risk contributions

  * var_*.py : historical, Cornish–Fisher and Monte Carlo VaR

pricing/ — Derivatives pricing and Monte Carlo simulations

* black_scholes.py : Black–Scholes model

* implied_vol.py : implied volatility computation and smile

* monte_carlo_gbm.py : GBM simulation

* asian_option_mc.py : Asian option pricing

* Hedging

  * delta_hedge_mc.py : Monte Carlo delta hedging

  * delta_hedge_backtest.py : hedging strategy backtesting

scripts/ — Reproducible execution scripts

* run_backtest.py : strategy backtesting execution

* run_var_mc.py : Monte Carlo risk simulations

* run_multiasset_frontier.py : multi-asset efficient frontiers

* run_vol_target.py : volatility targeting strategies

* other scripts for specific analyses

utils/ — Cross-cutting utilities

* data_loader.py : data loading and preprocessing

* walkforward.py : walkforward validation

* param_search.py : grid search and parameter tuning

* risk_mc.py : generic Monte Carlo utilities

data/ — Data and results

* market data (CSV)

* numerical results (CSV)

* figures and visualizations (PNG)

notebooks/ — Exploratory analyses (optional)

---

## Implemented Strategies

* **Moving Average Crossover (SMA)**
* **Trend Breakout**
* **Regime Filtering**
* **Volatility Targeting**
* **Risk Parity**
* **Minimum Variance**
* **Efficient Frontier**
* **Multi-asset allocation**
* **Walkforward optimization**
* **Stops & transaction costs**

---

## Risk Management

* **Value-at-Risk (VaR)** :

  * Historical
  * Cornish–Fisher
  * Monte Carlo
* **Expected Shortfall (ES)**
* **VaR backtests**
* **Risk contributions**
* **Monte Carlo PnL simulations**

---

## Pricing & Derivatives

* **Black–Scholes** model
* **Implied volatility** and smile
* **Monte Carlo (GBM)**
* **Asian options**
* **Delta hedging** (simulation and backtesting)
* Hedging error analysis

---

## Results and Visualizations

The project automatically generates:

* equity curves
* efficient frontiers
* parameter heatmaps
* risk distributions
* VaR backtests
* synthetic tear sheets

Results are exported in **CSV** and **PNG** formats for analysis and reporting.

---

## Reproducibility

* Dedicated scripts for each experiment
* Explicit parameters
* Fixed random seeds for Monte Carlo simulations
* Clear separation between **business logic** and **orchestration**

---

## Technical Environment

* **Python 3.13**
* NumPy, Pandas
* SciPy
* Matplotlib
* Scikit-learn
* Jupyter (exploration)

---

## Limitations and Perspectives

* Intentionally simple models (pedagogical and analytical objective)
* No market microstructure modeling
* No dynamic slippage modeling
* Possible extensions:

  * advanced stochastic models
  * liquidity constraints
  * robust optimization
  * integration with advanced statistical models

---

## Author

Personal project developed by a student in **M1 Applied Mathematics – Statistics**,
with strong interest in **quantitative finance**, **quant trading**, and **quantitative research**.

---

## License

Open-source project for educational purposes.