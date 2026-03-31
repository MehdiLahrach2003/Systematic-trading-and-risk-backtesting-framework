# backtesting/risk_contrib.py
# ------------------------------------------------------------
# Ce fichier contient des outils d'analyse de contribution au risque
# pour un portefeuille.
#
# Données d'entrée :
# - une matrice de covariance annualisée
# - un vecteur de poids
#
# Données de sortie :
# - volatilité totale du portefeuille
# - contribution marginale au risque (MRC)
# - contribution en composante au risque (CRC)
#
# Ces outils sont classiques pour :
# - le risk parity
# - l'analyse de portefeuille
# - comprendre quels actifs portent le risque total
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import math


@dataclass
class RiskContributionResult:
    """
    Objet de sortie contenant les résultats de contribution au risque.
    """
    weights: pd.Series   # poids du portefeuille
    vol_port: float      # volatilité totale du portefeuille
    mrc: pd.Series       # contribution marginale au risque
    crc: pd.Series       # contribution composante au risque


def risk_contributions(
    weights: pd.Series | np.ndarray,
    cov_ann: pd.DataFrame,
) -> RiskContributionResult:
    """
    Calcule la volatilité du portefeuille ainsi que les contributions au risque.

    Paramètres
    ----------
    weights : pd.Series ou np.ndarray
        Poids du portefeuille.
        Si c'est une Series, son index doit correspondre aux colonnes de cov_ann.
    cov_ann : pd.DataFrame
        Matrice de covariance annualisée des rendements.

    Retour
    ------
    RiskContributionResult
    """

    # ------------------------------------------------------------
    # 1) Transformer les poids en série bien alignée
    # ------------------------------------------------------------
    if isinstance(weights, pd.Series):
        w = weights.astype(float)
        assets = list(w.index)
    else:
        # Si les poids sont donnés sous forme de tableau numpy,
        # on suppose qu'ils sont dans le même ordre que les colonnes
        # de la matrice de covariance
        assets = list(cov_ann.columns)
        w = pd.Series(weights, index=assets, dtype=float)

    # ------------------------------------------------------------
    # 2) Réordonner la matrice de covariance selon l'ordre des actifs
    # ------------------------------------------------------------
    cov = cov_ann.loc[assets, assets].values
    w_vec = w.values

    # ------------------------------------------------------------
    # 3) Variance et volatilité du portefeuille
    # ------------------------------------------------------------
    # Formule :
    # variance portefeuille = w' Σ w
    var_port = float(w_vec @ cov @ w_vec)

    # Volatilité = racine carrée de la variance
    vol_port = math.sqrt(var_port) if var_port > 0.0 else 0.0

    # ------------------------------------------------------------
    # 4) Cas dégénéré : volatilité nulle
    # ------------------------------------------------------------
    if vol_port == 0.0:
        mrc = pd.Series(0.0, index=assets)
        crc = pd.Series(0.0, index=assets)
        return RiskContributionResult(weights=w, vol_port=0.0, mrc=mrc, crc=crc)

    # ------------------------------------------------------------
    # 5) Contribution marginale au risque (MRC)
    # ------------------------------------------------------------
    # Formule :
    # MRC_i = (Σ w)_i / sigma_portefeuille
    sigma_w = cov @ w_vec
    mrc_vec = sigma_w / vol_port

    # ------------------------------------------------------------
    # 6) Contribution composante au risque (CRC)
    # ------------------------------------------------------------
    # Formule :
    # CRC_i = w_i × MRC_i
    crc_vec = np.array(w_vec) * np.array(mrc_vec)

    # Mise sous forme de Series pandas
    mrc = pd.Series(mrc_vec, index=assets, name="MRC")
    crc = pd.Series(crc_vec, index=assets, name="CRC")

    return RiskContributionResult(
        weights=w,
        vol_port=vol_port,
        mrc=mrc,
        crc=crc,
    )