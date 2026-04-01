# scripts/run_var_cornish.py

"""
Comparaison entre :

- VaR gaussienne
- VaR Cornish–Fisher

pour la stratégie SMA 20/100.

Objectif :
voir si la prise en compte de l'asymétrie et de la kurtosis
modifie l'estimation du risque extrême.
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
from backtesting.engine import run_backtest
from backtesting.var_cornish import compute_cornish_report


def main():
    # --------------------------------------------------------
    # 1) Chargement des données
    # --------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # --------------------------------------------------------
    # 2) Construction de la stratégie SMA 20/100
    # --------------------------------------------------------
    pos = sma_crossover_positions(price, 20, 100)

    # --------------------------------------------------------
    # 3) Backtest
    # --------------------------------------------------------
    res = run_backtest(price, pos, cost_bps=1.0)

    # --------------------------------------------------------
    # 4) Calcul du rapport Cornish–Fisher
    # --------------------------------------------------------
    report = compute_cornish_report(res.returns, alpha=0.05)

    print("\n===== Gaussian vs Cornish–Fisher VaR (5%) =====")
    for k, v in report.items():
        print(f"{k:20s}: {v:.5f}")

    # --------------------------------------------------------
    # 5) Histogramme des rendements + seuils VaR
    # --------------------------------------------------------
    r = res.returns.dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(r, bins=50, alpha=0.6, label="Returns")

    var_g = report["VaR_gaussian"]
    var_cf = report["VaR_cornish"]

    # On trace les seuils en négatif car la VaR est stockée comme perte positive
    plt.axvline(-var_g, color="blue", linestyle="--", label=f"Gaussian VaR = {var_g:.4f}")
    plt.axvline(-var_cf, color="red", linestyle="--", label=f"Cornish–Fisher VaR = {var_cf:.4f}")

    plt.title("Gaussian vs Cornish–Fisher VaR (SMA 20/100)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "var_cornish_comparison.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()