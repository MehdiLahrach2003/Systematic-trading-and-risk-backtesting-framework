# backtesting/regime.py
# ------------------------------------------------------------
# Filtre de régime de marché.
#
# Ce fichier permet de dire :
# "est-ce que le marché est dans un bon état pour trader ?"
#
# Ici :
# - si le prix est au-dessus de sa moyenne long terme → régime = 1
# - sinon → régime = 0
# ------------------------------------------------------------

from dataclasses import dataclass
import pandas as pd


@dataclass
class RegimeConfig:
    """
    Paramètres du filtre de régime.

    long_window :
    taille de la fenêtre pour la moyenne long terme
    """
    long_window: int = 200


def long_only_regime(prices: pd.Series, config: RegimeConfig) -> pd.Series:
    """
    Construit un filtre de régime basé sur une moyenne mobile longue.

    Paramètres
    ----------
    prices : série de prix
    config : configuration (fenêtre de moyenne)

    Retour
    -------
    pd.Series avec valeurs :
    - 1 → marché favorable
    - 0 → marché défavorable
    """

    px = prices.astype(float)

    # Moyenne mobile long terme
    sma_long = px.rolling(config.long_window, min_periods=1).mean()

    # Régime : 1 si prix > moyenne, sinon 0
    regime = (px > sma_long).astype(float)
    regime.name = "regime"

    return regime