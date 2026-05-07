# pricing/asian_option_mc.py
# ------------------------------------------------------------
# Ce fichier permet de pricer une option asiatique par Monte Carlo.
#
# Particularité :
# le payoff d'une option asiatique dépend de la moyenne
# des prix du sous-jacent sur la trajectoire,
# et pas seulement du prix final S_T.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """
    Simule des trajectoires de prix selon un GBM.

    Paramètres
    ----------
    S0 : prix initial
    r : taux sans risque
    sigma : volatilité
    T : maturité
    n_steps : nombre de pas de temps
    n_paths : nombre de trajectoires
    seed : graine aléatoire

    Retour
    ------
    Tableau de taille (n_paths, n_steps + 1)
    contenant les trajectoires simulées.
    """
    np.random.seed(seed)

    dt = T / n_steps

    # Bruits gaussiens
    Z = np.random.normal(size=(n_paths, n_steps))

    # Incréments logarithmiques du GBM
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Construction des log-prix
    log_S = np.cumsum(increments, axis=1)

    # Construction des prix
    S = S0 * np.exp(log_S)

    # On ajoute le prix initial au début de chaque trajectoire
    S = np.hstack([np.full((n_paths, 1), S0), S])

    return S


def price_asian_option_mc(S0, K, r, sigma, T, n_steps, n_paths, option_type="call"):
    """
    Prix Monte Carlo d'une option asiatique à moyenne arithmétique.

    Paramètres
    ----------
    S0 : prix initial
    K : strike
    r : taux
    sigma : volatilité
    T : maturité
    n_steps : nombre de pas
    n_paths : nombre de trajectoires
    option_type : "call" ou "put"

    Retour
    ------
    price : prix estimé
    stderr : erreur standard
    """

    # 1) Simulation des trajectoires
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)

    # 2) Moyenne des prix sur chaque trajectoire
    # Ici on ne prend pas S0 dans la moyenne
    S_avg = np.mean(paths[:, 1:], axis=1)

    # 3) Payoff asiatique
    if option_type == "call":
        payoff = np.maximum(S_avg - K, 0)
    else:
        payoff = np.maximum(K - S_avg, 0)

    # 4) Actualisation
    discounted_payoff = np.exp(-r * T) * payoff

    # 5) Prix = moyenne des payoffs actualisés
    price = discounted_payoff.mean()

    # 6) Erreur standard Monte Carlo
    stderr = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)

    return price, stderr


def plot_sample_paths(S, T, n_show=10):
    """
    Affiche quelques trajectoires simulées.
    """
    n_steps = S.shape[1] - 1
    t = np.linspace(0, T, n_steps + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(t, S[:n_show].T, lw=1)
    plt.title("Trajectoires simulées pour l'option asiatique")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix du sous-jacent")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Paramètres de test
    S0 = 100
    K = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    n_steps = 252
    n_paths = 10000

    # Pricing Monte Carlo du call asiatique
    price, stderr = price_asian_option_mc(
        S0, K, r, sigma, T, n_steps, n_paths, "call"
    )

    print(f"Prix estimé du call asiatique : {price:.4f} ± {1.96*stderr:.4f} (IC 95%)")

    # Affichage de quelques trajectoires simulées
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, 50)
    plot_sample_paths(S, T)