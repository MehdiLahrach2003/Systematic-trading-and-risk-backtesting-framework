# pricing/monte_carlo_gbm.py
# ------------------------------------------------------------
# Ce fichier permet :
# - de simuler des trajectoires de prix selon un GBM
# - de pricer un call et un put européens par Monte Carlo
# - d'obtenir une erreur standard et un intervalle de confiance
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MCResult:
    """
    Résultat d'un pricing Monte Carlo.
    """
    price: float       # prix estimé
    std_err: float     # erreur standard
    ci_low: float      # borne basse IC 95%
    ci_high: float     # borne haute IC 95%
    n_paths: int       # nombre de chemins simulés


def simulate_gbm_paths(
    S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int, seed: int | None = 42
) -> np.ndarray:
    """
    Simule des trajectoires de prix selon un Geometric Brownian Motion.

    Retour :
    tableau de taille (steps+1, n_paths)
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / steps

    # Bruits gaussiens
    Z = rng.standard_normal(size=(steps, n_paths))

    # Incréments logarithmiques du GBM
    drift = (r - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * Z
    log_increments = drift + diff

    # Construction des log-prix puis des prix
    log_S = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
    S = S0 * np.exp(log_S)

    return S


def _discount(r: float, T: float) -> float:
    """
    Facteur d'actualisation.
    """
    return np.exp(-r * T)


def price_european_call_mc(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_paths: int = 100_000, steps: int = 1,
    antithetic: bool = True, seed: int | None = 42
) -> MCResult:
    """
    Prix d'un call européen par Monte Carlo.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Cas simple : on simule directement S_T
    if steps == 1:
        dt = T
        n_eff = n_paths

        if antithetic:
            # Réduction de variance : on prend Z et -Z
            Z = rng.standard_normal(n_paths // 2)
            Z = np.concatenate([Z, -Z])
            n_eff = Z.size
        else:
            Z = rng.standard_normal(n_paths)

        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    else:
        # Cas général : simulation complète de la trajectoire
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]
        n_eff = n_paths

    # Payoff du call
    payoff = np.maximum(ST - K, 0.0)

    # Actualisation
    disc = _discount(r, T)

    # Prix = moyenne actualisée des payoffs
    price = disc * payoff.mean()

    # Erreur standard + IC 95%
    std = payoff.std(ddof=1)
    std_err = disc * std / np.sqrt(n_eff)
    ci_low = price - 1.96 * std_err
    ci_high = price + 1.96 * std_err

    return MCResult(price, std_err, ci_low, ci_high, n_eff)


def price_european_put_mc(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_paths: int = 100_000, steps: int = 1,
    antithetic: bool = True, seed: int | None = 42
) -> MCResult:
    """
    Prix d'un put européen par Monte Carlo.
    """

    if steps == 1:
        rng = np.random.default_rng(seed)
        dt = T

        if antithetic:
            Z = rng.standard_normal(n_paths // 2)
            Z = np.concatenate([Z, -Z])
        else:
            Z = rng.standard_normal(n_paths)

        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        payoff = np.maximum(K - ST, 0.0)
        disc = _discount(r, T)

        price = disc * payoff.mean()
        std = payoff.std(ddof=1)
        std_err = disc * std / np.sqrt(ST.size)
        ci_low = price - 1.96 * std_err
        ci_high = price + 1.96 * std_err

        return MCResult(price, std_err, ci_low, ci_high, ST.size)

    else:
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]

        payoff = np.maximum(K - ST, 0.0) # type: ignore
        disc = _discount(r, T)

        price = disc * payoff.mean()
        std = payoff.std(ddof=1)
        std_err = disc * std / np.sqrt(n_paths)
        ci_low = price - 1.96 * std_err
        ci_high = price + 1.96 * std_err

        return MCResult(price, std_err, ci_low, ci_high, n_paths)


def demo_convergence(
    S0=100, K=100, r=0.03, sigma=0.2, T=1.0, steps=1, antithetic=True, seed=42
):
    """
    Trace la convergence du prix Monte Carlo quand le nombre de chemins augmente.
    """
    Ns = np.array([500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000])
    prices = []
    errs = []

    for n in Ns:
        res = price_european_call_mc(
            S0, K, r, sigma, T,
            n_paths=n,
            steps=steps,
            antithetic=antithetic,
            seed=seed
        )
        prices.append(res.price)
        errs.append(res.std_err)

    prices = np.array(prices)
    errs = np.array(errs)

    plt.figure(figsize=(7, 4))
    plt.plot(Ns, prices, marker='o', label="Prix MC (call)")
    plt.fill_between(Ns, prices - 1.96 * errs, prices + 1.96 * errs, alpha=0.2, label="IC 95%")
    plt.xscale("log")
    plt.xlabel("Nombre de chemins (log)")
    plt.ylabel("Prix")
    plt.title("Convergence Monte Carlo")
    plt.legend()
    plt.tight_layout()
    plt.show()


def demo_hist_terminal(
    S0=100, r=0.03, sigma=0.2, T=1.0, steps=1, n_paths=50_000, seed=42
):
    """
    Affiche l'histogramme de la distribution simulée de S_T.
    """
    if steps == 1:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    else:
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]

    plt.figure(figsize=(7, 4))
    plt.hist(ST, bins=60, density=True, alpha=0.7)
    plt.xlabel("S_T")
    plt.ylabel("Densité")
    plt.title("Distribution Monte Carlo de S_T")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0

    call_res = price_european_call_mc(S0, K, r, sigma, T, n_paths=100_000, steps=1, antithetic=True, seed=42)
    put_res  = price_european_put_mc(S0, K, r, sigma, T, n_paths=100_000, steps=1, antithetic=True, seed=42)

    print(f"Call MC: {call_res.price:.4f}  ± {1.96*call_res.std_err:.4f} (IC95%)  n={call_res.n_paths}")
    print(f"Put  MC: {put_res.price:.4f}  ± {1.96*put_res.std_err:.4f} (IC95%)  n={put_res.n_paths}")