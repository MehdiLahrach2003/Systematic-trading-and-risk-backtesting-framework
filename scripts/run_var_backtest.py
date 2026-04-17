# scripts/run_var_backtest.py
# ------------------------------------------------------------
# Ce script backteste une VaR historique glissante
# à l'aide du test de Kupiec.
#
# Objectif :
# vérifier si la VaR prévue est cohérente avec
# les pertes réellement observées.
# ------------------------------------------------------------

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.risk_measures import var_historical
from backtesting.var_backtest import kupiec_test


def compute_rolling_var(returns: pd.Series, window: int = 250, alpha: float = 0.05):
    """
    Calcule une VaR historique glissante.

    À chaque date :
    on prend les `window` rendements passés
    puis on calcule la VaR historique à niveau alpha.
    """
    var_series = returns.rolling(window).apply(
        lambda x: var_historical(pd.Series(x), alpha),
        raw=False
    )

    var_series.name = f"VaR_{int(alpha*100)}"
    return var_series


def main():
    # ------------------------------------------------------------
    # 1) Chargement des prix
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # ------------------------------------------------------------
    # 2) Construction de la stratégie SMA 20/100
    # ------------------------------------------------------------
    pos = sma_crossover_positions(price, 20, 100)

    # ------------------------------------------------------------
    # 3) Backtest de la stratégie
    # ------------------------------------------------------------
    res = run_backtest(price, pos, cost_bps=1.0)

    # ------------------------------------------------------------
    # 4) Calcul de la VaR historique glissante
    # ------------------------------------------------------------
    var_series = compute_rolling_var(res.returns, window=250, alpha=0.05)

    # ------------------------------------------------------------
    # 5) Test de Kupiec
    # ------------------------------------------------------------
    result = kupiec_test(res.returns, var_series, alpha=0.05)

    print("\n===== Kupiec Test (95% VaR) =====")
    for k, v in result.items():
        print(f"{k:10s}: {v}")

    # ------------------------------------------------------------
    # 6) Visualisation : rendements vs seuil de VaR
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    plt.plot(res.returns.index, res.returns, label="Returns", alpha=0.7)

    # On affiche -VaR car la VaR est stockée comme une perte positive
    plt.plot(var_series.index, -var_series, color="red", label="VaR (5%)")

    plt.title("Returns vs VaR (Rolling Historical)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Sauvegarde
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "var_backtest_plot.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()