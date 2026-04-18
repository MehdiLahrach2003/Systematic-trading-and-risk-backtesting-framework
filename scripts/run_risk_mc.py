# scripts/run_risk_mc.py
# ------------------------------------------------------------
# Analyse du risque par Monte Carlo pour la stratégie SMA 20/100.
#
# Idée :
# - on backteste d'abord la stratégie sur les données historiques
# - on récupère les rendements journaliers observés
# - on simule ensuite beaucoup de scénarios futurs
# - on regarde la distribution :
#   * de l'equity finale
#   * du drawdown maximal
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from utils.risk_mc import monte_carlo_from_returns


# ============================================================
# 1) Visualisation des résultats Monte Carlo
# ============================================================
def plot_mc_results(stats: pd.DataFrame) -> None:
    """
    Trace deux histogrammes :
    - distribution de l'equity finale
    - distribution du drawdown maximal
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --------------------------------------------------------
    # Histogramme de l'equity finale
    # --------------------------------------------------------
    ax1.hist(stats["final_equity"], bins=40, alpha=0.8)
    ax1.set_title("Distribution de l'equity finale")
    ax1.set_xlabel("Final equity (1.0 = capital initial)")
    ax1.set_ylabel("Fréquence")
    ax1.grid(alpha=0.3)

    # --------------------------------------------------------
    # Histogramme du max drawdown
    # --------------------------------------------------------
    ax2.hist(stats["max_drawdown"], bins=40, alpha=0.8)
    ax2.set_title("Distribution du drawdown maximal")
    ax2.set_xlabel("Max drawdown (négatif)")
    ax2.set_ylabel("Fréquence")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "risk_mc_histograms.png",
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Monte Carlo histograms saved -> {out_png}")

    plt.show()


# ============================================================
# 2) Main
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # 1) Backtest historique de la stratégie SMA 20/100
    # --------------------------------------------------------
    df = load_prices()
    price = df["price"]

    positions = sma_crossover_positions(price, short=20, long=100)

    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # --------------------------------------------------------
    # 2) Construction des rendements journaliers
    # à partir de la courbe d'equity
    # --------------------------------------------------------
    equity = result.equity
    returns = equity.pct_change().dropna()

    # --------------------------------------------------------
    # 3) Simulation Monte Carlo du risque
    # --------------------------------------------------------
    # On simule 2000 scénarios sur un horizon de 252 jours
    stats = monte_carlo_from_returns(
        returns,
        n_paths=2000,
        horizon=252,
        seed=42,
    )

    # --------------------------------------------------------
    # 4) Affichage de statistiques résumées
    # --------------------------------------------------------
    print("\n===== Monte Carlo risk (1-year horizon) =====")
    print(f"Mean final equity      : {stats['final_equity'].mean():.4f}")
    print(f"Median final equity    : {stats['final_equity'].median():.4f}")
    print(f"5% worst-case equity   : {stats['final_equity'].quantile(0.05):.4f}")
    print(f"1% worst-case equity   : {stats['final_equity'].quantile(0.01):.4f}")

    # Attention :
    # le drawdown est négatif, donc les quantiles se lisent différemment
    print(f"Median max drawdown    : {stats['max_drawdown'].median():.4f}")
    print(f"95% worst max drawdown : {stats['max_drawdown'].quantile(0.95):.4f}")

    # --------------------------------------------------------
    # 5) Tracé des histogrammes
    # --------------------------------------------------------
    plot_mc_results(stats)


if __name__ == "__main__":
    main()