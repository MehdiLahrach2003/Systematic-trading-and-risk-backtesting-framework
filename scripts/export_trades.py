# scripts/export_trades.py
# ------------------------------------------------------------
# Objectif :
# Exporter la liste des trades (entrées / sorties / P&L)
# générés par une stratégie (ici SMA 20/100) dans un fichier CSV.
# ------------------------------------------------------------

import os
import sys
import pandas as pd

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


def main():
    # --------------------------------------------------------
    # 1) Charger les données de prix
    # --------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # --------------------------------------------------------
    # 2) Construire les positions de la stratégie
    # SMA crossover 20/100
    # --------------------------------------------------------
    positions = sma_crossover_positions(price, short=20, long=100)

    # --------------------------------------------------------
    # 3) Lancer le backtest
    # (IMPORTANT : cost_bps = coûts de transaction)
    # --------------------------------------------------------
    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # --------------------------------------------------------
    # 4) Extraire les trades
    # result.trades = DataFrame avec toutes les opérations
    # --------------------------------------------------------
    trades = result.trades.copy()

    # Reset de l'index pour un CSV propre
    trades.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # 5) Sauvegarde en CSV
    # --------------------------------------------------------
    out_csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "trades_export.csv"
    )

    trades.to_csv(out_csv, index=False)

    # --------------------------------------------------------
    # 6) Affichage
    # --------------------------------------------------------
    print(f"\n[OK] Trades exported → {out_csv}\n")
    print(trades.head())


if __name__ == "__main__":
    main()