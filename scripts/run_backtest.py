# scripts/run_backtest.py
# ---------------------------------------------------------------------
# Ce script permet de lancer un backtest complet d'une stratégie simple :
# une stratégie de type "SMA crossover" (croisement de moyennes mobiles).
#
# IMPORTANT :
# Ce fichier n'est PAS le coeur mathématique du projet.
# Son rôle est d'orchestrer une expérience complète :
#
#   1) charger des données de prix
#   2) construire des positions de trading à partir d'une stratégie
#   3) exécuter le moteur de backtest
#   4) afficher les métriques de performance
#   5) tracer et sauvegarder les résultats
#
# C'est donc le meilleur point d'entrée pour comprendre
# comment le projet fonctionne concrètement.
# ---------------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce fichier directement
# ---------------------------------------------------------------------
# Quand on exécute ce script seul, Python ne sait pas forcément où trouver
# les dossiers "backtesting" et "utils".
#
# On ajoute donc manuellement le dossier racine du projet au path Python.
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------
# Import des briques principales utilisées dans ce script
# ---------------------------------------------------------------------

# 1) Chargement des données de prix
from utils.data_loader import load_prices

# 2) Construction des positions via une stratégie SMA crossover
from backtesting.ma_crossover import sma_crossover_positions

# 3) Moteur de backtest (transforme positions + prix en performance)
from backtesting.engine import run_backtest


def plot_backtest(df, positions, result):
    """
    Fonction qui trace les résultats du backtest.

    On affiche deux graphiques :
    1) Le prix + les signaux d'achat/vente
    2) La courbe de capital (equity) + les drawdowns

    Paramètres
    ----------
    df : DataFrame
        Contient les prix (colonne 'price')
    positions : Series
        Série des positions de trading (-1, 0, +1)
    result : BacktestResult
        Résultat du backtest (equity, returns, etc.)
    """

    # Création de la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}  # graphique du haut plus grand
    )

    # -----------------------------------------------------------------
    # 1) Graphique du prix + signaux
    # -----------------------------------------------------------------

    # Tracé du prix
    ax1.plot(df.index, df["price"], label="Price", color="black", lw=1.2)

    # Détection des signaux d'achat
    # On achète quand on passe de position <= 0 à > 0
    buys = positions[(positions.shift(1) <= 0) & (positions > 0)].index

    # Détection des signaux de vente
    # On vend quand on passe de position >= 0 à < 0
    sells = positions[(positions.shift(1) >= 0) & (positions < 0)].index

    # Affichage des points d'achat
    ax1.scatter(
        buys,
        df.loc[buys, "price"],
        color="green",
        marker="^",
        s=70,
        label="Buy"
    )

    # Affichage des points de vente
    ax1.scatter(
        sells,
        df.loc[sells, "price"],
        color="red",
        marker="v",
        s=70,
        label="Sell"
    )

    # Mise en forme du graphique
    ax1.set_title("SMA crossover — signaux de trading")
    ax1.set_ylabel("Prix")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # -----------------------------------------------------------------
    # 2) Courbe d'equity + drawdown
    # -----------------------------------------------------------------

    eq = result.equity  # capital au cours du temps

    # Courbe de capital
    ax2.plot(eq.index, eq, lw=1.6, label="Equity")

    # Zone de drawdown (distance au maximum historique)
    ax2.fill_between(
        eq.index,
        eq,
        eq.cummax(),
        color="red",
        alpha=0.12,
        label="Drawdown"
    )

    # Mise en forme
    ax2.set_title("Courbe de capital")
    ax2.set_ylabel("Capital")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Ajustement visuel
    plt.tight_layout()

    # Chemin de sauvegarde de l'image
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "backtest_plot.png"
    )

    # Création du dossier si nécessaire
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Sauvegarde du graphique
    plt.savefig(out_png, dpi=150)

    print(f"\n[OK] Graphique sauvegardé ici : {out_png}\n")

    # Affichage à l'écran
    plt.show()


def main():
    """
    Fonction principale : exécute tout le pipeline du backtest.
    """

    # -----------------------------------------------------------------
    # 1) Chargement des données
    # -----------------------------------------------------------------
    # load_prices() renvoie un DataFrame avec une colonne 'price'
    df = load_prices()

    # -----------------------------------------------------------------
    # 2) Construction des positions via la stratégie SMA
    # -----------------------------------------------------------------
    # Paramètres :
    # - moyenne courte = 20
    # - moyenne longue = 100
    #
    # La fonction renvoie des positions dans {-1, 0, +1}
    # et gère déjà le décalage pour éviter le lookahead bias.
    positions = sma_crossover_positions(df["price"], short=20, long=100)

    # -----------------------------------------------------------------
    # 3) Lancement du backtest
    # -----------------------------------------------------------------
    result = run_backtest(
        df["price"],
        positions,
        cost_bps=1.0  # coût de transaction (1 basis point)
    )

    # -----------------------------------------------------------------
    # 4) Affichage des métriques
    # -----------------------------------------------------------------
    print("\n===== Résultats du backtest =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")

    print("\nCapital final :", round(result.equity.iloc[-1], 4))

    # -----------------------------------------------------------------
    # 5) Visualisation
    # -----------------------------------------------------------------
    plot_backtest(df, positions, result)


# ---------------------------------------------------------------------
# Point d'entrée du script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()