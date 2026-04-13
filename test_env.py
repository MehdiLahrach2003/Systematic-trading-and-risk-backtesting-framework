""" 
Ce script sert à valider ton environnement de travail.

En gros : “Est-ce que tout est bien installé et prêt pour exécuter le projet ?
"""


import sys
import numpy as np

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Tout fonctionne !")