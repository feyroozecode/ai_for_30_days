# Jour 2 : Configuration pour Vos Premiers Modèles

## Vue d'ensemble

Le Jour 2 introduit vos premiers modèles d'apprentissage automatique utilisant scikit-learn. Nous construirons des modèles de régression linéaire et logistique.

## Prérequis

- Configuration du Jour 1 terminée
- Environnement virtuel activé
- Toutes les bibliothèques de `requirements.txt` installées

## Bibliothèques Requises

Le Jour 2 utilise scikit-learn, qui devrait déjà être installé depuis `requirements.txt`. Sinon, installez-le :

```bash
pip install scikit-learn
```

## Vérifier l'Installation

Exécutez ce code Python pour vérifier que scikit-learn fonctionne :

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification

# Tester la régression linéaire
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
model = LinearRegression()
model.fit(X, y)
print("Coefficients de régression linéaire:", model.coef_)
print("Intercept de régression linéaire:", model.intercept_)

# Tester la régression logistique
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
model = LogisticRegression()
model.fit(X, y)
print("Régression logistique entraînée avec succès !")

print("Toutes les bibliothèques du Jour 2 fonctionnent correctement !")
```

## Structure du Projet

Votre projet devrait maintenant inclure :

```
ai_bootcamp_env/
├── src/
│   ├── day01/
│   │   ├── data_utils.py
│   │   └── __init__.py
│   └── day02/
│       ├── models.py
│       └── __init__.py
├── tests/
│   ├── day01/
│   │   ├── test_data_utils.py
│   │   └── __init__.py
│   └── day02/
│       ├── test_models.py
│       └── __init__.py
├── docs/
│   ├── day01/
│   └── day02/
│       ├── concepts.md
│       └── setup.md
└── requirements.txt
```

## Exécuter le Code

### Tester les Modèles

```bash
# Exécuter les modèles directement
python src/day02/models.py
```

### Exécuter les Tests

```bash
# Exécuter tous les tests
pytest

# Exécuter seulement les tests du Jour 2
pytest tests/day02/
```

## Prochaines Étapes

Une fois la configuration terminée, vous pouvez :
1. Lire les concepts dans `docs/day02/concepts.md`
2. Exécuter le code d'exemple dans `src/day02/models.py`
3. Compléter les exercices en modifiant le code
4. Exécuter les tests pour vérifier vos implémentations

Bonne modélisation !