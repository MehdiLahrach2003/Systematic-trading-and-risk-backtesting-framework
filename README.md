# Quant Journey

**Quant Journey** est un projet personnel de recherche et de développement en **finance quantitative**, **gestion du risque** et **backtesting de stratégies**, développé dans le cadre de mon **M1 Mathématiques Appliquées – majeure Statistiques (Université Paris-Dauphine)**.

L’objectif est de construire un **framework quant cohérent et reproductible**, couvrant l’ensemble de la chaîne :
> données → stratégies → backtesting → gestion du risque → allocation de portefeuille → pricing de dérivés → analyse des résultats.

---

## 🎯 Objectifs du projet

- Implémenter un **moteur de backtesting** modulaire et extensible
- Étudier et comparer différentes **stratégies quantitatives**
- Mettre en œuvre des méthodes de **gestion du risque** (VaR, ES, contributions au risque)
- Explorer l’**allocation de portefeuille multi-actifs**
- Implémenter des modèles de **pricing et hedging de produits dérivés**
- Produire des **résultats analytiques et graphiques reproductibles**

Ce projet sert à la fois de **support d’apprentissage avancé** et de **portfolio technique** orienté quant / trading / research.

---

## 🧱 Architecture du projet

quant-journey/
│
├── backtesting/ # Moteur de backtest, stratégies et gestion du risque
│ ├── engine.py
│ ├── rules.py
│ ├── portfolio.py
│ ├── optimizer.py
│ ├── regime.py
│ ├── risk.py
│ ├── risk_measures.py
│ ├── risk_contrib.py
│ └── var_*.py
│
├── pricing/ # Pricing de dérivés et simulations Monte Carlo
│ ├── black_scholes.py
│ ├── implied_vol.py
│ ├── monte_carlo_gbm.py
│ ├── asian_option_mc.py
│ ├── delta_hedge_mc.py
│ └── delta_hedge_backtest.py
│
├── scripts/ # Scripts d’exécution reproductibles
│ ├── run_backtest.py
│ ├── run_var_mc.py
│ ├── run_multiaset_frontier.py
│ ├── run_vol_target.py
│ └── ...
│
├── utils/ # Chargement des données, walkforward, grid search
│ ├── data_loader.py
│ ├── walkforward.py
│ ├── param_search.py
│ └── risk_mc.py
│
├── data/ # Données, résultats, figures et métriques
│ ├── *.csv
│ └── *.png
│
├── notebooks/ # Analyses exploratoires (optionnel)
│
├── requirements.txt
├── README.md
└── LICENSE


---

## 📈 Stratégies implémentées

- **Moving Average Crossover (SMA)**
- **Trend Breakout**
- **Regime Filtering**
- **Volatility Targeting**
- **Risk Parity**
- **Minimum Variance**
- **Efficient Frontier**
- **Multi-asset allocation**
- **Walkforward optimization**
- **Stops & transaction costs**

---

## ⚠️ Gestion du risque

- **Value-at-Risk (VaR)** :
  - Historique
  - Cornish–Fisher
  - Monte Carlo
- **Expected Shortfall (ES)**
- **Backtests de VaR**
- **Contributions au risque**
- **Simulations Monte Carlo des PnL**

---

## 🧮 Pricing & dérivés

- Modèle **Black–Scholes**
- **Volatilité implicite** et smile
- **Monte Carlo (GBM)**
- **Options asiatiques**
- **Delta hedging** (simulation et backtest)
- Analyse des erreurs de couverture

---

## 📊 Résultats et visualisations

Le projet génère automatiquement :
- courbes de capital (equity curves)
- frontières efficientes
- heatmaps de paramètres
- distributions de risque
- backtests VaR
- tear sheets synthétiques

Les résultats sont exportés en **CSV** et **PNG** pour analyse et reporting.

---

## 🔁 Reproductibilité

- Scripts dédiés pour chaque expérience
- Paramètres explicites
- Seeds fixes pour les simulations Monte Carlo
- Séparation claire entre **logique métier** et **orchestration**

---

## 🛠️ Environnement technique

- **Python 3.13**
- NumPy, Pandas
- SciPy
- Matplotlib
- Scikit-learn
- Jupyter (exploration)

---

## 🚧 Limitations et perspectives

- Modèles volontairement simples (objectif pédagogique et analytique)
- Pas de microstructure de marché
- Pas de slippage dynamique
- Extensions possibles :
  - modèles stochastiques avancés
  - contraintes de liquidité
  - optimisation robuste
  - couplage avec modèles statistiques avancés

---

## 👤 Auteur

Projet personnel développé par un étudiant en **M1 Mathématiques Appliquées – Statistiques**,  
intéressé par la **finance quantitative**, le **quant trading** et la **recherche quantitative**.

---

## 📜 Licence

Projet open-source à but pédagogique.
