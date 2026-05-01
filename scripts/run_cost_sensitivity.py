# scripts/run_cost_sensitivity.py

"""
Analyse de sensibilité aux coûts de transaction pour la stratégie SMA 20/100.

Idée :
- charger les prix
- construire le signal SMA 20/100
- relancer le backtest pour plusieurs niveaux de coûts
- mesurer comment les performances se dégradent

C'est un script très important pour tester la robustesse réelle
d'une stratégie de trading.
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
from backtesting.engine import run_backtest


# ============================================================
# 1) Évaluation sur une grille de coûts
# ============================================================
def evaluate_cost_grid(
    df_prices: pd.DataFrame,
    cost_grid_bps: np.ndarray,
    short_window: int = 20,
    long_window: int = 100,
) -> pd.DataFrame:
    """
    Lance la stratégie SMA 20/100 pour plusieurs niveaux de coûts de transaction.

    Paramètres
    ----------
    df_prices : DataFrame
        doit contenir une colonne 'price'
    cost_grid_bps : array-like
        liste des coûts à tester, exprimés en basis points
    short_window : int
        fenêtre SMA courte
    long_window : int
        fenêtre SMA longue

    Retour
    ------
    DataFrame :
        index = niveau de coût
        colonnes = métriques du backtest
    """
    price = df_prices["price"].astype(float)

    # --------------------------------------------------------
    # On construit les positions UNE seule fois
    # La stratégie ne change pas, seuls les coûts changent
    # --------------------------------------------------------
    positions = sma_crossover_positions(price, short=short_window, long=long_window)

    rows = []

    for c_bps in cost_grid_bps:
        # ----------------------------------------------------
        # Backtest avec un niveau de coût donné
        # ----------------------------------------------------
        res = run_backtest(
            price_like=price,
            positions_like=positions,
            cost_bps=float(c_bps),
            initial_capital=1.0,
        )

        # Récupération des métriques
        m = res.metrics

        rows.append(
            {
                "cost_bps": float(c_bps),
                "Cumulative Return": m.get("Cumulative Return", np.nan),
                "Annualized Volatility": m.get("Annualized Volatility", np.nan),
                "Sharpe Ratio": m.get("Sharpe Ratio", np.nan),
                "Max Drawdown": m.get("Max Drawdown", np.nan),
                "Total Costs": m.get("Total Costs", np.nan),
                "Final Equity": float(res.equity.iloc[-1]),
            }
        )

    # Conversion en DataFrame final
    df_res = pd.DataFrame(rows).set_index("cost_bps").sort_index()
    return df_res


# ============================================================
# 2) Visualisation
# ============================================================
def plot_cost_sensitivity(df_res: pd.DataFrame, out_png: str | None = None) -> None:
    """
    Trace 4 graphes de sensibilité aux coûts :
    - Sharpe vs coûts
    - rendement cumulé vs coûts
    - max drawdown vs coûts
    - equity finale vs coûts
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = df_res.index.values

    # --------------------------------------------------------
    # 1) Sharpe ratio
    # --------------------------------------------------------
    ax1.plot(x, df_res["Sharpe Ratio"], marker="o")
    ax1.set_title("Sharpe vs coûts de transaction")
    ax1.set_xlabel("Coût (bps)")
    ax1.set_ylabel("Sharpe")
    ax1.grid(alpha=0.3)

    # --------------------------------------------------------
    # 2) Rendement cumulé
    # --------------------------------------------------------
    ax2.plot(x, df_res["Cumulative Return"], marker="o")
    ax2.set_title("Rendement cumulé vs coûts")
    ax2.set_xlabel("Coût (bps)")
    ax2.set_ylabel("Cumulative return")
    ax2.grid(alpha=0.3)

    # --------------------------------------------------------
    # 3) Max drawdown
    # --------------------------------------------------------
    ax3.plot(x, df_res["Max Drawdown"], marker="o")
    ax3.set_title("Max drawdown vs coûts")
    ax3.set_xlabel("Coût (bps)")
    ax3.set_ylabel("Max drawdown")
    ax3.grid(alpha=0.3)

    # --------------------------------------------------------
    # 4) Equity finale
    # --------------------------------------------------------
    ax4.plot(x, df_res["Final Equity"], marker="o")
    ax4.set_title("Equity finale vs coûts")
    ax4.set_xlabel("Coût (bps)")
    ax4.set_ylabel("Final equity")
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Cost-sensitivity plot saved → {out_png}")

    plt.show()


# ============================================================
# 3) Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    df = load_prices()

    # --------------------------------------------------------
    # 2) Grille de coûts en basis points
    # --------------------------------------------------------
    cost_grid_bps = np.array([0, 1, 2, 5, 10, 20, 30, 40, 50], dtype=float)

    # --------------------------------------------------------
    # 3) Analyse de sensibilité
    # --------------------------------------------------------
    df_res = evaluate_cost_grid(df, cost_grid_bps, short_window=20, long_window=100)

    # --------------------------------------------------------
    # 4) Sauvegarde des résultats dans un CSV
    # --------------------------------------------------------
    base_path = os.path.dirname(os.path.dirname(__file__))
    out_csv = os.path.join(base_path, "data", "cost_sensitivity_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv)
    print(f"[OK] Cost-sensitivity metrics saved → {out_csv}")

    # --------------------------------------------------------
    # 5) Tracé des résultats
    # --------------------------------------------------------
    out_png = os.path.join(base_path, "data", "cost_sensitivity_plot.png")
    plot_cost_sensitivity(df_res, out_png=out_png)

    # --------------------------------------------------------
    # 6) Affichage terminal
    # --------------------------------------------------------
    print("\n===== Cost-sensitivity summary (SMA 20/100) =====")
    print(df_res.round(4))


if __name__ == "__main__":
    main()