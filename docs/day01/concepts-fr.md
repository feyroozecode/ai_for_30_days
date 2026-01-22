# Jour 1 : Concepts d'apprentissage automatique

## Qu'est-ce que l'apprentissage automatique ?

L'apprentissage automatique (ML) est un sous-ensemble de l'intelligence artificielle qui permet aux ordinateurs d'apprendre et de prendre des décisions à partir de données sans être explicitement programmés pour chaque scénario.

## Types d'apprentissage automatique

### Apprentissage supervisé
- **Définition** : Apprendre à partir de données étiquetées pour prédire des résultats.
- **Exemples** : Classification (prédire des catégories), Régression (prédire des valeurs continues).
- **Cas d'usage** : Détection de spam dans les emails, prédiction des prix des maisons.

### Apprentissage non supervisé
- **Définition** : Trouver des patterns dans des données non étiquetées.
- **Exemples** : Clustering, Réduction de dimensionnalité.
- **Cas d'usage** : Segmentation des clients, détection d'anomalies.

## Le workflow ML

1. **Collecte de données** : Rassembler des données pertinentes
2. **Préparation des données** : Nettoyer, prétraiter et explorer les données
3. **Entraînement du modèle** : Entraîner des algorithmes ML sur des données préparées
4. **Évaluation du modèle** : Évaluer les performances du modèle sur des données invisibles
5. **Déploiement du modèle** : Intégrer le modèle dans des systèmes de production

## Outils clés pour la manipulation des données

### Pandas
- **Objectif** : Manipulation et analyse de données
- **Fonctionnalités clés** : DataFrames, Series, nettoyage de données, fusion
- **Exemple** :
  ```python
  import pandas as pd
  df = pd.read_csv('data.csv')
  print(df.head())
  ```

### NumPy
- **Objectif** : Calcul numérique
- **Fonctionnalités clés** : Tableaux, opérations mathématiques, algèbre linéaire
- **Exemple** :
  ```python
  import numpy as np
  arr = np.array([1, 2, 3, 4, 5])
  print(arr.mean())
  ```

### Matplotlib & Seaborn
- **Objectif** : Visualisation de données
- **Matplotlib** : Tracé de bas niveau
- **Seaborn** : Tracé statistique de haut niveau
- **Exemple** :
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Tracé simple
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

## Prochaines étapes

Avec ces concepts à l'esprit, pratiquez les bases avec les ressources fournies et commencez à écrire des tests de code simples pour les opérations de données.
