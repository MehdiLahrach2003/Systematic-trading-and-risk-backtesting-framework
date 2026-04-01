# scripts/run_var_mc.py
# ------------------------------------------------------------
# Ce script compare plusieurs méthodes d'estimation de la VaR
# et de la CVaR sur les rendements d'une stratégie.
#
# Méthodes comparées :
# - VaR gaussienne
# - VaR Monte Carlo paramétrique
# - VaR Monte Carlo bootstrap
# ------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.risk_measures import var_gaussian
from backtesting.var_montecarlo import mc_var_parametric, mc_var_bootstrap


def main():
    # ------------------------------------------------------------
    # 1) Chargement des prix + construction de la stratégie
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # Stratégie SMA 20/100
    pos = sma_crossover_positions(price, 20, 100)

    # Backtest
    res = run_backtest(price, pos, cost_bps=1.0)

    # Rendements de la stratégie
    ret = res.returns.dropna()

    # ------------------------------------------------------------
    # 2) VaR gaussienne
    # ------------------------------------------------------------
    var_g = var_gaussian(ret, alpha=0.05)

    # ------------------------------------------------------------
    # 3) Monte Carlo paramétrique
    # ------------------------------------------------------------
    var_mc_p, cvar_mc_p = mc_var_parametric(ret, alpha=0.05)

    # ------------------------------------------------------------
    # 4) Monte Carlo bootstrap
    # ------------------------------------------------------------
    var_mc_b, cvar_mc_b = mc_var_bootstrap(ret, alpha=0.05)

    # ------------------------------------------------------------
    # 5) Affichage des résultats numériques
    # ------------------------------------------------------------
    print("\n===== VaR / CVaR Comparison (5%) =====")
    print(f"Gaussian VaR        : {var_g:.5f}")
    print(f"Parametric MC  VaR  : {var_mc_p:.5f}   |  CVaR = {cvar_mc_p:.5f}")
    print(f"Bootstrap   MC  VaR : {var_mc_b:.5f}   |  CVaR = {cvar_mc_b:.5f}")

    # ------------------------------------------------------------
    # 6) Histogramme des rendements + seuils VaR
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # Distribution empirique des rendements
    plt.hist(ret, bins=50, alpha=0.6, label="Returns")

    # Seuils de VaR (placés à -VaR car la VaR est positive comme perte)
    plt.axvline(-var_g, color="blue", linestyle="--", label=f"Gaussian VaR = {var_g:.4f}")
    plt.axvline(-var_mc_p, color="red", linestyle="--", label=f"MC Parametric = {var_mc_p:.4f}")
    plt.axvline(-var_mc_b, color="purple", linestyle="--", label=f"MC Bootstrap = {var_mc_b:.4f}")

    plt.title("Histogram of returns with VaR thresholds (5%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Sauvegarde
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "var_mc_comparison.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()