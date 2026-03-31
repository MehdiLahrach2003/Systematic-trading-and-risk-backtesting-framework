# backtesting/rules.py
# ------------------------------------------------------------
# Ce fichier ajoute des règles de trading réalistes :
# - stop-loss
# - take-profit
#
# Il modifie une stratégie de base pour limiter les pertes
# et sécuriser les gains.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StopConfig:
    """
    Paramètres du stop-loss et du take-profit.

    Exemple :
    - 0.05 = -5% (stop-loss)
    - 0.10 = +10% (take-profit)
    """
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


def apply_stop_loss_take_profit(
    prices: pd.Series,
    base_positions: pd.Series,
    config: StopConfig,
) -> pd.Series:

    prices = prices.astype(float)
    pos_raw = base_positions.reindex(prices.index).fillna(0.0)

    stop_loss = config.stop_loss_pct
    take_profit = config.take_profit_pct

    # On va reconstruire les positions une par une
    pos = pos_raw.copy()
    pos.values[:] = 0.0

    in_pos = 0.0          # position actuelle (-1, 0, +1)
    entry_price = None    # prix d'entrée

    for i, (dt, price) in enumerate(prices.items()):

        desired = float(pos_raw.iloc[i])

        # ------------------------------------------------------------
        # 1) Si on n'a pas de position → on peut entrer
        # ------------------------------------------------------------
        if in_pos == 0.0:
            if desired != 0.0:
                in_pos = desired
                entry_price = float(price)

            pos.iloc[i] = in_pos
            continue

        # ------------------------------------------------------------
        # 2) Si on est en position → calcul du rendement
        # ------------------------------------------------------------
        ret_since_entry = price / entry_price - 1.0

        # Stop-loss
        hit_stop = (
            stop_loss is not None
            and ret_since_entry <= -abs(stop_loss)
        )

        # Take-profit
        hit_tp = (
            take_profit is not None
            and ret_since_entry >= abs(take_profit)
        )

        # ------------------------------------------------------------
        # 3) Si stop ou take-profit → sortie
        # ------------------------------------------------------------
        if hit_stop or hit_tp:
            in_pos = 0.0
            entry_price = None
            pos.iloc[i] = 0.0
            continue

        # ------------------------------------------------------------
        # 4) Sinon → suivre le signal de base
        # ------------------------------------------------------------
        if desired == 0.0:
            in_pos = 0.0
            entry_price = None

        elif desired != in_pos:
            # inversion de position
            in_pos = desired
            entry_price = float(price)

        pos.iloc[i] = in_pos

    pos.name = "position_with_stops"
    return pos