# scripts/run_regime_filter.py
# ------------------------------------------------------------
# Ce script compare une stratégie SMA crossover :
# - sans filtre de régime
# - avec un filtre de régime long terme
#
# Objectif :
# voir si la stratégie marche mieux lorsqu'on ne l'active
# que dans les phases de marché jugées favorables.
# ------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.regime import RegimeConfig, long_only_regime


def plot_equity_comparison(base_res, filtered_res):
    """
    Trace les courbes d'equity de :
    - la stratégie SMA de base
    - la stratégie SMA filtrée par le régime
    """
    # Rebase à 1 pour comparer proprement
    eq_base = base_res.equity / base_res.equity.iloc[0]
    eq_filt = filtered_res.equity / filtered_res.equity.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(eq_base.index, eq_base, label="SMA 20/100 (no filter)", lw=1.4)
    plt.plot(eq_filt.index, eq_filt, label="SMA 20/100 + regime filter", lw=1.4)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity: SMA crossover with vs without regime filter")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Sauvegarde de la figure
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "regime_equity.png",
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
    # 1) Chargement des prix
    # ------------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # ------------------------------------------------------------
    # 2) Construction de la stratégie SMA de base
    # ------------------------------------------------------------
    base_pos = sma_crossover_positions(price, short=20, long=100)

    # ------------------------------------------------------------
    # 3) Construction du filtre de régime long terme
    # ------------------------------------------------------------
    # Ici, le régime est défini par une moyenne mobile 200 périodes :
    # - prix > SMA 200 -> régime favorable (1)
    # - sinon -> régime défavorable (0)
    cfg = RegimeConfig(long_window=200)
    regime = long_only_regime(price, cfg)

    # ------------------------------------------------------------
    # 4) Application du filtre
    # ------------------------------------------------------------
    # Si régime = 1, on garde la position SMA
    # Si régime = 0, la position devient nulle
    filt_pos = base_pos * regime
    filt_pos.name = "position_filtered"

    # ------------------------------------------------------------
    # 5) Backtest de la stratégie de base
    # ------------------------------------------------------------
    base_res = run_backtest(
        df,
        base_pos,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # ------------------------------------------------------------
    # 6) Backtest de la stratégie filtrée
    # ------------------------------------------------------------
    filt_res = run_backtest(
        df,
        filt_pos,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # ------------------------------------------------------------
    # 7) Affichage des métriques
    # ------------------------------------------------------------
    print_metrics("SMA 20/100 (no filter)", base_res)
    print_metrics("SMA 20/100 + regime filter", filt_res)

    # ------------------------------------------------------------
    # 8) Comparaison visuelle des equity curves
    # ------------------------------------------------------------
    plot_equity_comparison(base_res, filt_res)


if __name__ == "__main__":
    main()