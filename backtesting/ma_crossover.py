# backtesting/ma_crossover.py
# ---------------------------------------------------------
# Stratégie SMA crossover :
# on compare une moyenne courte et une moyenne longue
# pour décider d’être long (+1), short (-1) ou flat (0)
# ---------------------------------------------------------

import pandas as pd


def sma_crossover_positions(prices: pd.Series, short: int = 20, long: int = 100) -> pd.Series:
    """
    Construit une série de positions de trading.

    Entrée :
        prices = série de prix

    Sortie :
        positions dans {-1, 0, +1}
    """

    # On s'assure que les prix sont bien en float
    px = prices.astype(float)

    # Moyenne mobile courte (réagit vite)
    sma_short = px.rolling(short, min_periods=1).mean()

    # Moyenne mobile longue (plus lente)
    sma_long = px.rolling(long, min_periods=1).mean()

    # Signal brut :
    # +1 si courte > longue
    # -1 si courte < longue
    #  0 sinon
    raw_signal = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)

    # Décalage d’un jour (TRÈS IMPORTANT)
    # → on utilise l’info d’hier pour trader aujourd’hui
    positions = raw_signal.shift(1).fillna(0.0)

    positions.name = "position"
    return positions