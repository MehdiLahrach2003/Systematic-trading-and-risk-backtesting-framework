# scripts/run_multiasset_corr.py
# ------------------------------------------------------------
# Ce script compare :
# - la corrélation entre les rendements bruts des actifs
# - la corrélation entre les rendements des stratégies SMA 20/100
#
# Objectif :
# voir si la transformation "actif -> stratégie"
# modifie la structure de corrélation,
# ce qui est crucial pour la diversification.
# ------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ------------------------------------------------------------
# 1) Rendements logarithmiques des actifs
# ------------------------------------------------------------
def compute_asset_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques journaliers
    de chaque actif.
    """
    log_px = np.log(df_prices)
    returns = log_px.diff().dropna() # type: ignore
    return returns


# ------------------------------------------------------------
# 2) Rendements des stratégies SMA sur chaque actif
# ------------------------------------------------------------
def compute_sma_strategy_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque actif :
    - construit une stratégie SMA 20/100
    - lance le backtest
    - récupère les rendements journaliers de stratégie
    """
    strat_returns = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # Signal SMA
        pos = sma_crossover_positions(price, short=20, long=100)

        # Backtest
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        # Rendements nets de stratégie
        strat_returns[symbol] = res.returns

    # Alignement sur un index commun
    strat_ret_df = pd.concat(strat_returns, axis=1).dropna()
    strat_ret_df.columns = df_prices.columns
    return strat_ret_df


# ------------------------------------------------------------
# 3) Heatmap de corrélation
# ------------------------------------------------------------
def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_name: str) -> None:
    """
    Trace et sauvegarde une heatmap de corrélation.
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr_df.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")

    plt.title(title)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right") # type: ignore
    plt.yticks(range(len(corr_df.index)), corr_df.index) # type: ignore

    # Valeurs numériques dans chaque case
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.7 else "white", # type: ignore
                fontsize=9,
            )

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        out_name
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Correlation heatmap saved → {out_png}")

    plt.show()


# ------------------------------------------------------------
# 4) Script principal
# ------------------------------------------------------------
def main():
    # Univers d'actifs
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # Chargement des prix multi-actifs
    df_prices = load_multi_assets(symbols)
    df_prices = df_prices[symbols].dropna(how="all")

    # Corrélation des rendements bruts
    asset_rets = compute_asset_returns(df_prices)
    asset_corr = asset_rets.corr()

    print("\n===== Correlation matrix – asset returns =====")
    print(asset_corr.round(3))

    plot_corr_heatmap(
        corr_df=asset_corr,
        title="Correlation of asset daily returns",
        out_name="multiasset_price_corr.png",
    )

    # Corrélation des rendements de stratégies SMA
    strat_rets = compute_sma_strategy_returns(df_prices)
    strat_corr = strat_rets.corr()

    print("\n===== Correlation matrix – SMA 20/100 strategy returns =====")
    print(strat_corr.round(3))

    plot_corr_heatmap(
        corr_df=strat_corr,
        title="Correlation of SMA 20/100 strategy returns",
        out_name="multiasset_strategy_corr.png",
    )


if __name__ == "__main__":
    main()