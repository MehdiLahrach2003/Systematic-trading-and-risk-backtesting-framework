# scripts/run_portfolio_risk_parity.py

"""
Risk-parity portfolio entre deux stratégies :

1) SMA 20/100 (trend following classique)
2) Breakout 50 jours

Objectif :
→ Construire un portefeuille où CHAQUE stratégie contribue autant au risque
→ Utiliser une approximation simple : poids ∝ 1 / volatilité

Ce script :
- backteste chaque stratégie
- estime leur volatilité
- construit les poids risk parity
- combine les returns
- affiche les résultats + plot
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Permet d'importer les modules du projet (backtesting, utils, etc.)
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions


# ============================================================
# 1) Construction des stratégies
# ============================================================
def build_strategies(df, cost_bps: float = 1.0):
    """
    Construit et backteste deux stratégies :

    - SMA 20/100 → signal trend classique
    - Breakout 50 jours → signal de momentum

    Retourne :
    - résultats du backtest pour chaque stratégie
    """

    price = df["price"]

    # --- Stratégie 1 : SMA crossover ---
    sma_pos = sma_crossover_positions(price, short=20, long=100)

    # run_backtest :
    # → transforme les positions en returns + equity
    sma_res = run_backtest(price, sma_pos, cost_bps=cost_bps, initial_capital=1.0)

    # --- Stratégie 2 : Breakout ---
    brk_pos = breakout_positions(price, lookback=50)

    brk_res = run_backtest(price, brk_pos, cost_bps=cost_bps, initial_capital=1.0)

    return sma_res, brk_res


# ============================================================
# 2) Calcul des poids risk parity
# ============================================================
def compute_risk_parity_weights(sma_res, brk_res):
    """
    Approximation simple du risk parity :

    poids ∝ 1 / volatilité

    → une stratégie volatile reçoit moins de poids
    → une stratégie stable reçoit plus de poids
    """

    # returns journaliers
    r_sma = sma_res.returns.dropna()
    r_brk = brk_res.returns.dropna()

    # volatilités (écart-type)
    vol_sma = r_sma.std()
    vol_brk = r_brk.std()

    # Cas dégénéré (rare mais safe coding)
    if vol_sma == 0 or vol_brk == 0:
        print("[WARN] Volatilité nulle → fallback 50/50")
        return 0.5, 0.5

    # Inverse volatility
    inv_vol_sma = 1.0 / vol_sma
    inv_vol_brk = 1.0 / vol_brk

    # Normalisation → somme = 1
    total = inv_vol_sma + inv_vol_brk
    w_sma = inv_vol_sma / total
    w_brk = inv_vol_brk / total

    return float(w_sma), float(w_brk)


# ============================================================
# 3) Construction du portefeuille
# ============================================================
def build_portfolio_returns(sma_res, brk_res, w_sma: float, w_brk: float):
    """
    Combine les returns journaliers :

    r_portfolio = w_sma * r_sma + w_brk * r_brk
    """

    import pandas as pd

    # On aligne les dates
    idx = sma_res.returns.index.intersection(brk_res.returns.index)

    r_sma = sma_res.returns.reindex(idx)
    r_brk = brk_res.returns.reindex(idx)

    # Combinaison linéaire
    portfolio_ret = w_sma * r_sma + w_brk * r_brk
    portfolio_ret.name = "portfolio_ret"

    return portfolio_ret


# ============================================================
# 4) Statistiques du portefeuille
# ============================================================
def summarize_portfolio(portfolio_ret):
    """
    Calcule :

    - rendement cumulé
    - volatilité annualisée
    - Sharpe ratio
    """

    r = portfolio_ret.dropna()

    if r.empty:
        return {"Cumulative Return": 0.0, "Annualized Vol": 0.0, "Sharpe": 0.0}

    # rendement total
    cum_ret = float((1.0 + r).prod() - 1.0)

    # volatilité annualisée
    ann_vol = float(r.std() * np.sqrt(252))

    # Sharpe ratio
    if ann_vol == 0:
        sharpe = 0.0
    else:
        sharpe = float(r.mean() / r.std() * np.sqrt(252))

    return {
        "Cumulative Return": cum_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
    }


# ============================================================
# 5) Visualisation
# ============================================================
def plot_equity_curves(sma_res, brk_res, portfolio_ret, w_sma: float, w_brk: float):
    """
    Plot :

    - equity SMA
    - equity Breakout
    - equity portefeuille risk parity
    """

    import pandas as pd

    # Construction equity du portefeuille
    eq_port = (1.0 + portfolio_ret).cumprod()

    # Rebase à 1
    eq_sma = sma_res.equity / sma_res.equity.iloc[0]
    eq_brk = brk_res.equity / brk_res.equity.iloc[0]
    eq_port = eq_port / eq_port.iloc[0]

    # Alignement
    idx_all = eq_sma.index.union(eq_brk.index).union(eq_port.index)
    eq_sma = eq_sma.reindex(idx_all)
    eq_brk = eq_brk.reindex(idx_all)
    eq_port = eq_port.reindex(idx_all)

    plt.figure(figsize=(10, 6))

    plt.plot(eq_sma.index, eq_sma, label="SMA 20/100", lw=1.2)
    plt.plot(eq_brk.index, eq_brk, label="Breakout 50d", lw=1.2)

    plt.plot(
        eq_port.index,
        eq_port,
        label=f"Risk-parity ({w_sma:.0%} SMA / {w_brk:.0%} Breakout)",
        lw=1.8,
    )

    plt.axhline(1.0, color="black", lw=0.8, ls="--")

    plt.title("Risk-parity portfolio vs individual strategies")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# 6) MAIN
# ============================================================
def main():
    # 1) Chargement des données
    df = load_prices()

    # 2) Backtests individuels
    sma_res, brk_res = build_strategies(df)

    # 3) Poids risk parity
    w_sma, w_brk = compute_risk_parity_weights(sma_res, brk_res)

    print("\n===== Risk-parity weights =====")
    print(f"SMA weight      : {w_sma:.4f}")
    print(f"Breakout weight : {w_brk:.4f}")

    # 4) Construction portefeuille
    portfolio_ret = build_portfolio_returns(sma_res, brk_res, w_sma, w_brk)

    # 5) Stats
    stats = summarize_portfolio(portfolio_ret)

    print("\n===== Portfolio metrics =====")
    for k, v in stats.items():
        print(f"{k:20s}: {v:.4f}")

    # 6) Plot
    plot_equity_curves(sma_res, brk_res, portfolio_ret, w_sma, w_brk)


if __name__ == "__main__":
    main()