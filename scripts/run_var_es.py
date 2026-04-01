# scripts/run_var_es.py

"""
Calcule la Value-at-Risk (VaR) et l'Expected Shortfall (CVaR)
pour la stratégie SMA 20/100.

Objectif :
- mesurer les pertes extrêmes
- comparer un simple seuil (VaR) à la moyenne des pires pertes (CVaR)
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
from backtesting.risk_measures import compute_risk_measures


def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # --------------------------------------------------------
    # 2) Construction de la stratégie SMA 20/100
    # --------------------------------------------------------
    pos = sma_crossover_positions(price, short=20, long=100)

    # --------------------------------------------------------
    # 3) Backtest
    # --------------------------------------------------------
    res = run_backtest(price, pos, cost_bps=1.0)

    # --------------------------------------------------------
    # 4) Calcul des mesures de risque à 5%
    # --------------------------------------------------------
    metrics = compute_risk_measures(res.returns, alpha=0.05)

    print("\n===== Risk Measures (5% tail) =====")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.5f}")

    print(f"\nFinal equity: {res.equity.iloc[-1]:.4f}")

    # --------------------------------------------------------
    # 5) Histogramme des rendements + seuil de VaR historique
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    r = res.returns.dropna()
    plt.hist(r, bins=50, alpha=0.6, label="Returns")

    var_hist = metrics["VaR_hist"]

    # On trace -VaR car la VaR est stockée comme une perte positive
    plt.axvline(
        -var_hist,
        color="red",
        linestyle="--",
        label=f"VaR 5% = {var_hist:.4f}"
    )

    plt.title("Distribution des rendements avec seuil de VaR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "var_es_histogram.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Histogram saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()