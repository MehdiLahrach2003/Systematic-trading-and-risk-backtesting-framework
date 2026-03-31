# backtesting/risk.py
# ------------------------------------------------------------
# Ce fichier contient des outils simples de gestion du risque.
#
# Contrairement à risk_measures.py qui MESURE le risque
# (VaR, CVaR, etc.), ici on agit plutôt sur la stratégie
# ou on calcule des objets utiles liés au risque :
#
# - volatility targeting
# - equity curve à partir des rendements
# - drawdown
# - maximum drawdown
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1) Volatility targeting
# ------------------------------------------------------------
def vol_target_positions(
    returns: pd.Series,
    positions: pd.Series,
    target_vol_annual: float = 0.10,   # volatilité annuelle cible, par exemple 10%
    lookback: int = 20,                # fenêtre glissante pour estimer la vol
    ann_factor: int = 252,             # nombre de jours de bourse par an
    max_leverage: float = 3.0,         # borne maximale sur l'exposition
) -> pd.Series:
    """
    Ajuste la taille des positions pour viser une volatilité cible.

    Idée :
    - si la volatilité observée est forte, on réduit l'exposition
    - si la volatilité observée est faible, on augmente l'exposition

    Paramètres
    ----------
    returns : pd.Series
        Rendements journaliers du sous-jacent.
    positions : pd.Series
        Positions brutes (par exemple -1, 0, +1), déjà décalées
        pour éviter le lookahead bias.
    target_vol_annual : float
        Volatilité annuelle cible de la stratégie.
    lookback : int
        Taille de la fenêtre glissante pour estimer la volatilité.
    ann_factor : int
        Facteur d'annualisation.
    max_leverage : float
        Exposition maximale autorisée en valeur absolue.

    Retour
    ------
    pd.Series
        Série d'expositions ajustées.
    """

    # On nettoie les rendements
    ret = returns.astype(float).fillna(0.0)

    # Volatilité glissante journalière
    vol_daily = ret.rolling(lookback).std()

    # Volatilité annualisée
    vol_annual = vol_daily * np.sqrt(ann_factor)

    # Facteur d'ajustement = volatilité cible / volatilité observée
    scale = target_vol_annual / vol_annual.replace(0.0, np.nan)

    # On borne le levier maximal
    scale = scale.clip(upper=max_leverage)

    # On enlève les infinis / NaN éventuels
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Nouvelle exposition = position brute × facteur d'ajustement
    expo = positions.astype(float) * scale
    expo.name = "exposure"

    # Important :
    # on ne décale pas ici, car on suppose que les positions
    # ont déjà été décalées au niveau du signal
    return expo


# ------------------------------------------------------------
# 2) Equity curve à partir des rendements
# ------------------------------------------------------------
def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """
    Transforme une série de rendements en courbe de capital.

    Paramètres
    ----------
    returns : pd.Series
        Rendements journaliers (ou périodiques).
    initial : float
        Capital initial.

    Retour
    ------
    pd.Series
        Courbe d'equity.
    """

    # Nettoyage
    ret = pd.Series(returns).astype(float).fillna(0.0)

    # Capital cumulé
    eq = initial * (1.0 + ret).cumprod()
    eq.name = "equity"
    return eq


# ------------------------------------------------------------
# 3) Drawdown
# ------------------------------------------------------------
def drawdown(equity: pd.Series) -> pd.Series:
    """
    Calcule la série de drawdown à partir d'une courbe d'equity.

    Le drawdown mesure la perte relative par rapport
    au maximum historique atteint jusque-là.

    Paramètres
    ----------
    equity : pd.Series
        Courbe de capital.

    Retour
    ------
    pd.Series
        Série de drawdown (toujours <= 0).
    """

    # Nettoyage
    eq = pd.Series(equity).astype(float)

    # Maximum historique courant
    peak = eq.cummax()

    # Drawdown = écart relatif au maximum passé
    dd = eq / peak - 1.0
    dd.name = "drawdown"
    return dd


# ------------------------------------------------------------
# 4) Maximum drawdown
# ------------------------------------------------------------
def max_drawdown(equity: pd.Series) -> float:
    """
    Calcule le maximum drawdown d'une courbe d'equity.

    Paramètres
    ----------
    equity : pd.Series
        Courbe de capital.

    Retour
    ------
    float
        Plus mauvais drawdown observé.
        Exemple : -0.32 signifie une perte max de 32%.
    """

    dd = drawdown(equity)
    return float(dd.min())