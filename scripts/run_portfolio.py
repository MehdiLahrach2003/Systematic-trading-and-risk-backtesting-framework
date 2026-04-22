# scripts/run_portfolio.py

"""
Exemple simple de portefeuille entre deux stratégies :

- SMA crossover 20/100
- Breakout 50 jours

Objectif :
- backtester chaque stratégie séparément
- les combiner dans un portefeuille 50/50
- comparer les performances individuelles et la performance du portefeuille
"""

import os
import sys

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions
from backtesting.engine import run_backtest
from backtesting.portfolio import combine_backtests


# ============================================================
# 1) Tracé des courbes d'equity
# ============================================================
def plot_equity_comparison(results_dict):
    """
    Trace les courbes d'equity de plusieurs objets BacktestResult.

    Paramètre
    ---------
    results_dict : dict
        dictionnaire de la forme :
        nom_stratégie -> BacktestResult
    """
    plt.figure(figsize=(10, 6))

    for name, res in results_dict.items():
        # Rebase à 1 pour comparer facilement les trajectoires
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity curves – stratégies individuelles vs portefeuille")
    plt.ylabel("Equity (rebasée à 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "portfolio_equity.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Portfolio equity plot saved → {out_png}")

    plt.show()


# ============================================================
# 2) Affichage des métriques
# ============================================================
def print_metrics(title, res):
    """
    Affiche proprement les métriques d'un BacktestResult.
    """
    print(f"\n===== {title} =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")


# ============================================================
# 3) Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    # Si un CSV existe dans data/, il est utilisé.
    # Sinon, le loader peut générer une série synthétique.
    df = load_prices()
    price = df["price"]

    # --------------------------------------------------------
    # 2) Construction des positions des deux stratégies
    # --------------------------------------------------------

    # Stratégie 1 : SMA crossover 20/100
    sma_pos = sma_crossover_positions(price, short=20, long=100)

    # Stratégie 2 : Breakout 50 jours
    brk_pos = breakout_positions(price, lookback=50)

    # --------------------------------------------------------
    # 3) Backtest de chaque stratégie séparément
    # --------------------------------------------------------
    sma_res = run_backtest(df, sma_pos, cost_bps=1.0, initial_capital=1.0)
    brk_res = run_backtest(df, brk_pos, cost_bps=1.0, initial_capital=1.0)

    # --------------------------------------------------------
    # 4) Construction d'un portefeuille 50/50
    # --------------------------------------------------------
    results = {
        "SMA 20/100": sma_res,
        "Breakout 50d": brk_res,
    }

    weights = {
        "SMA 20/100": 0.5,
        "Breakout 50d": 0.5,
    }

    # Combine les deux BacktestResult en un seul portefeuille
    portfolio_res = combine_backtests(results, weights=weights)

    # On ajoute le portefeuille au dictionnaire pour l'affichage/plot
    results["Portfolio 50/50"] = portfolio_res

    # --------------------------------------------------------
    # 5) Affichage des métriques
    # --------------------------------------------------------
    print_metrics("SMA 20/100", sma_res)
    print_metrics("Breakout 50d", brk_res)
    print_metrics("Portfolio 50/50", portfolio_res)

    # --------------------------------------------------------
    # 6) Tracé comparatif des courbes d'equity
    # --------------------------------------------------------
    plot_equity_comparison(results)


if __name__ == "__main__":
    main()