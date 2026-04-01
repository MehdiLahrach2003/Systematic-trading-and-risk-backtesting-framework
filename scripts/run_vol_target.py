# scripts/run_vol_target.py
# ------------------------------------------------------------
# Ce script exécute une stratégie SMA crossover
# avec une couche de volatility targeting.
#
# Objectif :
# ne pas seulement décider du sens du trade,
# mais aussi ajuster la taille de la position
# en fonction de la volatilité récente du marché.
# ------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.risk import vol_target_positions
from backtesting.engine import run_backtest


def plot_equity(result, title="Vol-targeted equity"):
    """
    Trace la courbe d'equity de la stratégie
    avec une zone visuelle de drawdown.
    """
    plt.figure(figsize=(11, 5))

    eq = result.equity

    # Courbe de capital
    plt.plot(eq.index, eq.values, label="Equity", lw=1.6)

    # Zone de drawdown
    plt.fill_between(eq.index, eq.values, eq.cummax(), alpha=0.12, color="red")

    plt.title(title)
    plt.ylabel("Capital")
    plt.grid(alpha=0.3)
    plt.legend()

    # Sauvegarde de la figure
    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "vol_target_equity.png"
    )

    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"[OK] Equity plot saved -> {out}")
    plt.show()


def main():
    # ------------------------------------------------------------
    # 1) Chargement des prix
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # Rendements du sous-jacent
    returns = price.pct_change().fillna(0.0)

    # ------------------------------------------------------------
    # 2) Construction de la stratégie SMA brute
    # ------------------------------------------------------------
    # Positions dans {-1, 0, +1}
    pos_raw = sma_crossover_positions(price, short=20, long=100)

    # ------------------------------------------------------------
    # 3) Ajout du volatility targeting
    # ------------------------------------------------------------
    # La stratégie garde la même direction,
    # mais la taille de la position est ajustée
    # pour viser une volatilité annuelle de 10%.
    pos_vt = vol_target_positions(
        returns=returns,
        positions=pos_raw,
        target_vol_annual=0.10,   # cible de volatilité annuelle = 10%
        lookback=20,              # estimation sur les 20 dernières périodes
        max_leverage=3.0,         # levier maximal autorisé
    )

    # ------------------------------------------------------------
    # 4) Backtest de la stratégie ajustée
    # ------------------------------------------------------------
    res = run_backtest(df, pos_vt, cost_bps=1.0)

    # ------------------------------------------------------------
    # 5) Affichage des métriques
    # ------------------------------------------------------------
    print("\n===== Vol Target Backtest (SMA 20/100) =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")

    print(f"\nFinal equity: {res.equity.iloc[-1]:.2f}")

    # ------------------------------------------------------------
    # 6) Tracé de la courbe d'equity
    # ------------------------------------------------------------
    plot_equity(res, title="SMA 20/100 with Vol Target (10%)")


if __name__ == "__main__":
    main()