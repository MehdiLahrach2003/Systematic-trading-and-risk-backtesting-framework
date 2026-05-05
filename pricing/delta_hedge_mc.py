# pricing/delta_hedge_mc.py
# ------------------------------------------------------------
# Ce fichier étudie l'erreur de couverture en delta
# d'un call européen dans le modèle de Black-Scholes.
#
# Idée :
# - on simule des trajectoires GBM du sous-jacent
# - sur chaque trajectoire, on couvre dynamiquement le call
#   avec son delta
# - à la fin, on compare la valeur du portefeuille
#   de réplication au payoff réel de l'option
# ------------------------------------------------------------

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import du prix du call et du delta Black-Scholes
try:
    from .black_scholes import bs_call_price, bs_call_delta
except ImportError:
    from black_scholes import bs_call_price, bs_call_delta  # type: ignore


# ------------------------------------------------------------
# 1) Une seule trajectoire + couverture en delta
# ------------------------------------------------------------
def simulate_delta_hedge_path(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simule une trajectoire GBM et applique une couverture en delta
    d'un call européen tout au long de la trajectoire.

    Retour :
    - times : grille de temps
    - spot : trajectoire simulée du sous-jacent
    - portfolio : valeur du portefeuille de réplication
    - hedging_error : erreur finale de couverture
    """

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # ------------------------------------------------------------
    # 1) Simulation d'une trajectoire du sous-jacent
    # ------------------------------------------------------------
    times = np.linspace(0.0, T, n_steps + 1)

    spot = np.empty(n_steps + 1, dtype=float)
    spot[0] = S0

    for i in range(1, n_steps + 1):
        z = rng.normal()
        spot[i] = spot[i - 1] * math.exp(
            (r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z
        )

    # ------------------------------------------------------------
    # 2) Mise en place du portefeuille de couverture
    # ------------------------------------------------------------
    portfolio = np.empty(n_steps + 1, dtype=float)

    # Prix initial du call
    V0 = bs_call_price(S0, K, r, sigma, T)

    # Delta initial
    delta0 = bs_call_delta(S0, K, r, sigma, T)

    # On détient delta0 actions
    shares = delta0

    # Le reste est en cash
    cash = V0 - shares * S0

    # Valeur initiale du portefeuille = prix du call
    portfolio[0] = V0

    # ------------------------------------------------------------
    # 3) Rééquilibrage dynamique
    # ------------------------------------------------------------
    for i in range(1, n_steps + 1):
        t_i = times[i]
        S_i = spot[i]

        # Le cash grossit au taux sans risque
        cash *= math.exp(r * dt)

        # Valeur du portefeuille juste avant rééquilibrage
        portfolio[i] = shares * S_i + cash

        # Tant qu'on n'est pas à maturité, on met à jour le delta
        if i < n_steps:
            tau = T - t_i
            new_delta = bs_call_delta(S_i, K, r, sigma, tau)

            # Variation du nombre d'actions
            d_shares = new_delta - shares

            # Achat/vente financé par le cash
            cash -= d_shares * S_i

            # Nouveau nombre d'actions
            shares = new_delta

    # ------------------------------------------------------------
    # 4) Erreur finale de couverture
    # ------------------------------------------------------------
    payoff_T = max(spot[-1] - K, 0.0)
    portfolio_T = portfolio[-1]

    hedging_error = portfolio_T - payoff_T

    return times, spot, portfolio, hedging_error


# ------------------------------------------------------------
# 2) Répéter sur beaucoup de trajectoires
# ------------------------------------------------------------
def simulate_hedging_errors_mc(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 52,
    n_paths: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Lance une simulation Monte Carlo des erreurs de couverture.

    Retour :
    - errors : vecteur des erreurs finales sur tous les chemins
    """
    rng = np.random.default_rng(seed)
    errors = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        _, _, _, err = simulate_delta_hedge_path(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            rng=rng,
        )
        errors[i] = err

    return errors


# ------------------------------------------------------------
# 3) Démo complète
# ------------------------------------------------------------
def main():
    # Paramètres
    S0 = 100.0
    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_steps = 52
    n_paths = 10_000

    # Simulation des erreurs
    errors = simulate_hedging_errors_mc(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=123,
    )

    # Statistiques simples
    mean_err = float(errors.mean())
    std_err = float(errors.std(ddof=1))
    q05, q50, q95 = np.quantile(errors, [0.05, 0.5, 0.95])

    print("\n=== Delta-hedging error Monte Carlo ===")
    print(f"Paths           : {n_paths}")
    print(f"Rebalancing     : {n_steps} steps over T = {T}y")
    print(f"Mean error      : {mean_err:.6f}")
    print(f"Std of error    : {std_err:.6f}")
    print(f"5% / 50% / 95%  : {q05:.6f}  {q50:.6f}  {q95:.6f}")

    # Histogramme des erreurs
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.axvline(0.0, color="black", lw=1.2, ls="--", label="Couverture parfaite")
    plt.title("Distribution de l'erreur de delta-hedging")
    plt.xlabel("Erreur de couverture")
    plt.ylabel("Densité")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()