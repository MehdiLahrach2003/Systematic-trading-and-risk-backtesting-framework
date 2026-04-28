# scripts/run_breakout.py
# ------------------------------------------------------------
# Ce script exécute une stratégie breakout
# et affiche ses résultats.
#
# Il fait :
# - chargement des prix
# - construction des positions breakout
# - backtest
# - affichage des métriques
# - visualisation du prix et de l'equity
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.engine import run_backtest
from backtesting.trend_breakout import breakout_positions


def plot_breakout_equity(df, positions, result):
    """
    Trace :
    - le prix avec un fond coloré selon la position
    - la courbe d'equity rebasée à 1
    """
    price = df["price"]

    # Rebase de l'equity à 1 pour lecture plus simple
    eq = result.equity / result.equity.iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ------------------------------------------------------------
    # 1) Prix + coloration par type de position
    # ------------------------------------------------------------
    ax1.plot(price.index, price, color="black", lw=1.2, label="Price")
    ax1.set_title("Breakout strategy — price & positions")

    # Masques de position
    long_mask = positions > 0
    short_mask = positions < 0
    flat_mask = positions == 0

    # Zone verte = long
    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=long_mask,
        color="green",
        alpha=0.08,
        label="Long",
    )

    # Zone rouge = short
    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=short_mask,
        color="red",
        alpha=0.08,
        label="Short",
    )

    # Zone grise = flat
    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=flat_mask,
        color="grey",
        alpha=0.04,
        label="Flat",
    )

    ax1.legend()
    ax1.grid(alpha=0.3)

    # ------------------------------------------------------------
    # 2) Courbe d'equity
    # ------------------------------------------------------------
    ax2.plot(eq.index, eq, lw=1.5, label="Breakout equity")
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_title("Equity curve (rebased to 1.0)")
    ax2.set_ylabel("Index")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Sauvegarde de la figure
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "breakout_equity.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Breakout equity figure saved → {out_png}")
    plt.show()


def main():
    # ------------------------------------------------------------
    # 1) Chargement des prix
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # ------------------------------------------------------------
    # 2) Construction des positions breakout
    # ------------------------------------------------------------
    positions = breakout_positions(
        price,
        lookback=80,   # fenêtre de breakout
        hold_bars=5,   # durée minimale de détention
    )

    # ------------------------------------------------------------
    # 3) Backtest de la stratégie breakout
    # ------------------------------------------------------------
    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # ------------------------------------------------------------
    # 4) Affichage des métriques
    # ------------------------------------------------------------
    print("\n===== Breakout strategy results =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {result.equity.iloc[-1]:.4f}")

    # ------------------------------------------------------------
    # 5) Tracé des résultats
    # ------------------------------------------------------------
    plot_breakout_equity(df, positions, result)


if __name__ == "__main__":
    main()