# scripts/run_portfolio_grid.py

"""
Recherche sur grille des poids de portefeuille entre deux stratégies :

- SMA 20/100
- Breakout 50 jours

Objectif :
- faire varier le poids de la stratégie SMA entre 0% et 100%
- construire le portefeuille correspondant
- mesurer ses performances
- visualiser le compromis rendement / risque
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions
from backtesting.engine import run_backtest, combine_backtests


# ============================================================
# 1) Backtests des stratégies individuelles
# ============================================================
def run_individual_strategies(df, cost_bps: float = 1.0):
    """
    Lance les deux stratégies de base :
    - SMA 20/100
    - Breakout 50 jours

    Retour :
    dict contenant les BacktestResult des deux stratégies
    """
    price = df["price"]

    # --- SMA 20/100 ---
    sma_pos = sma_crossover_positions(price, short=20, long=100)
    sma_res = run_backtest(
        df,
        sma_pos,
        cost_bps=cost_bps,
        initial_capital=1.0,
    )

    # --- Breakout 50 jours ---
    brk_pos = breakout_positions(price, lookback=50)
    brk_res = run_backtest(
        df,
        brk_pos,
        cost_bps=cost_bps,
        initial_capital=1.0,
    )

    results = {
        "sma_20_100": sma_res,
        "breakout_50": brk_res,
    }
    return results


# ============================================================
# 2) Construction d'une grille de portefeuilles
# ============================================================
def portfolio_grid(results, n_steps: int = 11):
    """
    Construit une grille de portefeuilles entre SMA et Breakout.

    Exemple si n_steps = 11 :
    - 0% SMA / 100% Breakout
    - 10% SMA / 90% Breakout
    - ...
    - 100% SMA / 0% Breakout

    Retour :
    DataFrame avec poids + métriques de performance
    """
    sma_key = "sma_20_100"
    brk_key = "breakout_50"

    # Poids possibles pour SMA
    weights_sma = np.linspace(0.0, 1.0, n_steps)

    rows = []

    for w_sma in weights_sma:
        w_brk = 1.0 - w_sma

        weights = {
            sma_key: w_sma,
            brk_key: w_brk,
        }

        # Combinaison des deux stratégies dans un portefeuille
        port_res = combine_backtests(results, weights)

        # On copie les métriques du portefeuille
        m = dict(port_res.metrics)

        # On ajoute les poids et l'equity finale
        m["w_sma_20_100"] = w_sma
        m["w_breakout_50"] = w_brk
        m["Final equity"] = float(port_res.equity.iloc[-1])

        rows.append(m)

    df_grid = pd.DataFrame(rows)

    # Réorganisation des colonnes
    cols = [
        "w_sma_20_100",
        "w_breakout_50",
        "Cumulative Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Total Costs",
        "Final equity",
    ]
    df_grid = df_grid[[c for c in cols if c in df_grid.columns]]

    return df_grid


# ============================================================
# 3) Visualisation rendement / risque
# ============================================================
def plot_risk_return(df_grid: pd.DataFrame, out_png: str | None = None):
    """
    Trace le nuage rendement cumulé vs volatilité annualisée
    pour tous les portefeuilles de la grille.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = df_grid["Annualized Volatility"]
    y = df_grid["Cumulative Return"]

    ax.scatter(x, y, alpha=0.8)

    # Annotation de quelques portefeuilles clés
    for _, row in df_grid.iterrows():
        w = row["w_sma_20_100"]
        label = None

        if abs(w - 0.0) < 1e-8:
            label = "0% SMA / 100% Breakout"
        elif abs(w - 0.5) < 1e-8:
            label = "50% / 50%"
        elif abs(w - 1.0) < 1e-8:
            label = "100% SMA / 0% Breakout"

        if label is not None:
            ax.annotate(
                label,
                (row["Annualized Volatility"], row["Cumulative Return"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Grille rendement / risque – SMA 20/100 vs Breakout 50d")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Frontier plot saved → {out_png}")

    plt.show()


# ============================================================
# 4) Main
# ============================================================
def main():
    # 1) Chargement des données de prix
    df = load_prices()

    # 2) Backtests des deux stratégies
    results = run_individual_strategies(df, cost_bps=1.0)

    # 3) Construction de la grille de portefeuilles
    df_grid = portfolio_grid(results, n_steps=11)

    # 4) Sauvegarde des résultats dans un CSV
    root = os.path.dirname(os.path.dirname(__file__))
    out_csv = os.path.join(root, "data", "portfolio_grid_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_grid.to_csv(out_csv, index=False)
    print(f"[OK] Portfolio grid results saved → {out_csv}")

    # 5) Tracé rendement / risque
    out_png = os.path.join(root, "data", "portfolio_frontier.png")
    plot_risk_return(df_grid, out_png=out_png)


if __name__ == "__main__":
    main()