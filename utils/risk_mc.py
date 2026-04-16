# utils/risk_mc.py

# ------------------------------------------------------------
# Monte Carlo simple basé sur les rendements historiques
#
# Objectif :
# - simuler des trajectoires futures de la stratégie
# - estimer les risques (drawdown, perte, etc.)
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.risk import max_drawdown


def monte_carlo_from_returns(
    returns: pd.Series,
    n_paths: int = 2000,
    horizon: int = 252,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Monte Carlo par bootstrap des rendements.

    Principe :
    - on tire aléatoirement (avec remise) des rendements passés
    - on reconstruit une trajectoire d'equity
    - on mesure ses propriétés

    Paramètres
    ----------
    returns : pd.Series
        rendements journaliers de la stratégie
    n_paths : int
        nombre de scénarios simulés
    horizon : int
        nombre de jours simulés (ex: 252 = 1 an)
    seed : int | None
        graine aléatoire

    Retour
    -------
    DataFrame avec :
        - final_equity
        - max_drawdown
    """

    # --------------------------------------------------------
    # 1) Nettoyage des rendements
    # --------------------------------------------------------
    ret = returns.dropna().to_numpy()

    if ret.size == 0:
        raise ValueError("returns series is empty – cannot run Monte Carlo.")

    # Générateur aléatoire
    rng = np.random.default_rng(seed)

    # Tableaux pour stocker les résultats
    final_equity = np.empty(n_paths)
    max_dd = np.empty(n_paths)

    # --------------------------------------------------------
    # 2) Boucle Monte Carlo
    # --------------------------------------------------------
    for i in range(n_paths):

        # Tirage aléatoire avec remise
        sample = rng.choice(ret, size=horizon, replace=True)

        # Construction de la trajectoire d'equity :
        # equity_t = produit des (1 + returns)
        equity_path = pd.Series(
            np.cumprod(1.0 + sample),
            index=range(horizon)
        )

        # Valeur finale
        final_equity[i] = equity_path.iloc[-1]

        # Max drawdown de cette trajectoire
        max_dd[i] = max_drawdown(equity_path)

    # --------------------------------------------------------
    # 3) Retour des résultats
    # --------------------------------------------------------
    return pd.DataFrame(
        {
            "final_equity": final_equity,
            "max_drawdown": max_dd,
        }
    )