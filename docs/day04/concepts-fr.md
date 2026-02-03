# Jour 4 : Optimisation des Hyperparamètres & Sélection de Modèles

## Que sont les Hyperparamètres ?

Les **hyperparamètres** sont des variables de configuration qui contrôlent le processus d'apprentissage d'un modèle de machine learning. Contrairement aux paramètres du modèle (qui sont appris à partir des données), les hyperparamètres sont définis avant le début de l'entraînement.

### Exemples d'Hyperparamètres

| Modèle | Hyperparamètres | Description |
|--------|----------------|-------------|
| Régression Logistique | `C`, `penalty` | Force et type de régularisation |
| Forêt Aléatoire | `n_estimators`, `max_depth` | Nombre d'arbres et leur profondeur |
| Ridge/Lasso | `alpha` | Force de régularisation |
| Réseaux de Neurones | `learning_rate`, `batch_size` | Configuration d'entraînement |

### Paramètres vs Hyperparamètres

- **Paramètres** : Appris à partir des données pendant l'entraînement (ex: `coef_`, `intercept_`)
- **Hyperparamètres** : Définis par vous avant l'entraînement (ex: `max_depth`, `C`)

## Pourquoi Optimiser les Hyperparamètres ?

Les hyperparamètres par défaut donnent rarement les meilleures performances. L'optimisation permet de :

1. **Améliorer les Performances** : Trouver la configuration optimale pour votre dataset spécifique
2. **Prévenir le Surapprentissage** : Contrôler la complexité du modèle (ex: limiter la profondeur des arbres)
3. **Prévenir le Sous-apprentissage** : Permettre suffisamment de complexité pour capturer les patterns
4. **Optimiser le Temps d'Entraînement** : Équilibrer précision et coût computationnel

## Grid Search (Recherche par Grille)

### Concept

Le **Grid Search** teste exhaustivement chaque combinaison d'hyperparamètres à partir d'une grille prédéfinie.

### Fonctionnement

1. Définir une grille de valeurs d'hyperparamètres à tester
2. Pour chaque combinaison :
   - Entraîner le modèle en utilisant la validation croisée
   - Calculer la performance moyenne
3. Retourner la combinaison avec le meilleur score

### Exemple

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Définir la grille de paramètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}

# Créer le modèle
model = RandomForestClassifier(random_state=42)

# Effectuer la recherche par grille
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Meilleurs paramètres
print(f"Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur score : {grid_search.best_score_:.4f}")
```

### Avantages & Inconvénients

**Avantages :**
- Garantit de trouver la meilleure combinaison dans la grille
- Systématique et exhaustif

**Inconvénients :**
- Coûteux en calcul (croissance exponentielle avec plus de paramètres)
- Limité aux valeurs discrètes que vous spécifiez

## Random Search (Recherche Aléatoire)

### Concept

Le **Random Search** échantillonne aléatoirement des combinaisons d'hyperparamètres à partir de distributions spécifiées.

### Fonctionnement

1. Définir des distributions pour les hyperparamètres (listes ou distributions statistiques)
2. Échantillonner aléatoirement `n_iter` combinaisons
3. Évaluer chacune en utilisant la validation croisée
4. Retourner la meilleure combinaison trouvée

### Exemple

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Définir les distributions de paramètres
param_dist = {
    'n_estimators': randint(50, 300),  # Entier aléatoire entre 50-300
    'max_depth': [3, 5, 7, 10, None]
}

# Effectuer la recherche aléatoire
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=5, random_state=42
)
random_search.fit(X, y)
```

### Quand Utiliser la Recherche Aléatoire

- Espaces de recherche d'hyperparamètres larges
- Budget computationnel limité
- Hyperparamètres continus (ex: taux d'apprentissage)

### Insight de Recherche

[Bergstra & Bengio (2012)](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) ont montré que la recherche aléatoire est souvent plus efficace que la recherche par grille, surtout quand seulement quelques hyperparamètres affectent significativement les performances.

## Comparaison de Modèles

### Pourquoi Comparer les Modèles ?

Différents algorithmes font différentes hypothèses sur les données. Comparer plusieurs modèles permet de :
- Identifier quel algorithme fonctionne le mieux pour votre problème
- Comprendre les compromis entre précision et interprétabilité
- Construire des modèles d'ensemble utilisant les meilleurs performers

### Workflow de Comparaison

```python
from sklearn.model_selection import cross_val_score

models = {
    'Régression Logistique': LogisticRegression(),
    'Forêt Aléatoire': RandomForestClassifier(),
    'SVM': SVC()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
```

## Pipelines ML

### Le Problème : Fuite de Données

Quand on prétraite les données (mise à l'échelle, imputation) avant de les séparer :
- L'information du jeu de test fuit dans l'entraînement
- Conduit à des estimations de performance trop optimistes

### La Solution : Les Pipelines

Les **Pipelines** garantissent que les étapes de prétraitement sont ajustées uniquement sur les données d'entraînement pendant la validation croisée.

### Exemple de Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Créer le pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Étape 1 : Mise à l'échelle
    ('classifier', LogisticRegression())    # Étape 2 : Entraînement
])

# Utiliser dans la recherche par grille
param_grid = {
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)
```

### Avantages des Pipelines

1. **Prévenir la Fuite de Données** : Prétraitement ajusté uniquement sur les plis d'entraînement
2. **Code Plus Propre** : Un seul objet pour toutes les étapes
3. **Déploiement Plus Facile** : Sérialiser un objet au lieu de plusieurs
4. **API Cohérente** : Même interface `fit()`, `predict()`

## Bonnes Pratiques

### 1. Commencer Simple
Commencer avec une petite grille et étendre en fonction des résultats

### 2. Utiliser la Validation Croisée
Toujours utiliser la CV lors de l'optimisation pour éviter le surapprentissage sur un ensemble de validation spécifique

### 3. Choisir la Métrique Appropriée
- Classification : `accuracy`, `f1`, `roc_auc`, `precision`, `recall`
- Régression : `r2`, `neg_mean_squared_error`, `neg_mean_absolute_error`

### 4. Surveiller l'Écart Entraînement vs Test
Les grands écarts indiquent un surapprentissage - simplifier le modèle

### 5. Journaliser Vos Expériences
Suivre les hyperparamètres et résultats pour la reproductibilité

### 6. Considérer le Temps de Calcul
- Grid Search : Teste toutes les combinaisons (lent mais exhaustif)
- Random Search : Teste un nombre fixe (plus rapide, souvent aussi bon)

## Grilles d'Hyperparamètres Communs

### Régression Logistique
```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

### Forêt Aléatoire
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### Régression Ridge
```python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
```

## Prochaines Étapes

Demain, nous appliquerons tout ce que nous avons appris pour construire un projet ML complet - le défi de Prédiction de Survie du Titanic. Cela intégrera :
- Chargement et prétraitement des données
- Entraînement de modèles avec optimisation des hyperparamètres
- Évaluation appropriée avec validation croisée
- Comparaison et sélection de modèles
