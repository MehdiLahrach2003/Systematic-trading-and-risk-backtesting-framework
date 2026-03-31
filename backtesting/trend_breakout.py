# backtesting/trend_breakout.py
# ------------------------------------------------------------
# Stratégie de breakout (suivi de tendance).
#
# Idée :
# - acheter si le prix dépasse le max récent
# - vendre si le prix casse le min récent
# - sinon garder la position
# ------------------------------------------------------------

from __future__ import annotations
import pandas as pd


def breakout_positions(
    prices: pd.Series,
    lookback: int = 50,
    hold_bars: int = 5,
) -> pd.Series:

    # Vérification du type
    if not isinstance(prices, pd.Series):
        raise TypeError("`prices` must be a pandas Series.")

    px = prices.astype(float)

    # ------------------------------------------------------------
    # 1) Calcul des extrêmes passés
    # ------------------------------------------------------------
    # shift(1) = on exclut le jour courant (pas de triche)
    rolling_high = px.shift(1).rolling(lookback, min_periods=lookback // 2).max()
    rolling_low = px.shift(1).rolling(lookback, min_periods=lookback // 2).min()

    # ------------------------------------------------------------
    # 2) Détection des breakouts
    # ------------------------------------------------------------
    long_signal = px > rolling_high
    short_signal = px < rolling_low

    # +1 si breakout haut, -1 si breakout bas, 0 sinon
    raw = long_signal.astype(int) - short_signal.astype(int)

    # ------------------------------------------------------------
    # 3) Gestion de position avec contrainte de durée minimale
    # ------------------------------------------------------------
    pos = pd.Series(0.0, index=px.index, name="position")

    current_pos = 0.0
    bars_since_change = 0

    for i, (idx, sig) in enumerate(raw.items()):
        bars_since_change += 1

        # Si nouveau signal ET on a respecté la durée minimale
        if sig != 0 and (bars_since_change >= hold_bars or current_pos == 0):

            # Changement de position
            if sig != current_pos:
                current_pos = sig
                bars_since_change = 0

        pos.iloc[i] = current_pos

    # ------------------------------------------------------------
    # 4) Décalage pour éviter le look-ahead
    # ------------------------------------------------------------
    pos = pos.shift(1).fillna(0.0)

    return pos