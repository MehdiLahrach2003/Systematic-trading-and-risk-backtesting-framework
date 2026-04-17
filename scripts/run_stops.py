# scripts/run_stops.py
# ------------------------------------------------------------
# Ce script compare une stratégie SMA crossover :
# - sans stops
# - avec stop-loss et take-profit
#
# Objectif :
# voir si l'ajout de règles de protection améliore
# la performance ou réduit le risque.
# ------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.rules import StopConfig, apply_stop_loss_take_profit


def plot_equity_comparison(base_res, stopped_res):
    """
    Trace les courbes d'equity de :
    - la stratégie de base
    - la stratégie avec stops
    """
    # Rebase à 1 pour comparer plus facilement
    eq_base = base_res.equity / base_res.equity.iloc[0]
    eq_stop = stopped_res.equity / stopped_res.equity.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(eq_base.index, eq_base, label="SMA 20/100 (no stops)", lw=1.4)
    plt.plot(eq_stop.index, eq_stop, label="SMA 20/100 + stops", lw=1.4)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity curve: SMA crossover with vs without stops")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Sauvegarde de la figure
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "stops_equity.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Equity comparison saved → {out_png}")
    plt.show()


def print_metrics(title: str, result):
    """
    Affiche proprement les métriques du backtest.
    """
    print(f"\n===== {title} =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {result.equity.iloc[-1]:.4f}")


def main():
    # ------------------------------------------------------------
    # 1) Chargement des données
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # ------------------------------------------------------------
    # 2) Construction de la stratégie de base
    # ------------------------------------------------------------
    base_positions = sma_crossover_positions(price, short=20, long=100)

    # ------------------------------------------------------------
    # 3) Backtest de la stratégie sans stops
    # ------------------------------------------------------------
    base_res = run_backtest(
        price,
        base_positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # ------------------------------------------------------------
    # 4) Définition des paramètres de stops
    # ------------------------------------------------------------
    cfg = StopConfig(
        stop_loss_pct=0.05,    # couper si perte de 5%
        take_profit_pct=0.10,  # sortir si gain de 10%
    )

    # ------------------------------------------------------------
    # 5) Application des stops à la stratégie
    # ------------------------------------------------------------
    stopped_positions = apply_stop_loss_take_profit(
        prices=price,
        base_positions=base_positions,
        config=cfg,
    )

    # ------------------------------------------------------------
    # 6) Backtest de la stratégie avec stops
    # ------------------------------------------------------------
    stopped_res = run_backtest(
        price,
        stopped_positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # ------------------------------------------------------------
    # 7) Comparaison des métriques
    # ------------------------------------------------------------
    print_metrics("SMA 20/100 (no stops)", base_res)
    print_metrics("SMA 20/100 + stops", stopped_res)

    # ------------------------------------------------------------
    # 8) Comparaison visuelle
    # ------------------------------------------------------------
    plot_equity_comparison(base_res, stopped_res)


if __name__ == "__main__":
    main()