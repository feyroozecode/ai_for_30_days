# Jour 2 : Vos Premiers Modèles - Régression Linéaire & Logistique

## Qu'est-ce qu'une "Feature" et une "Target" ?

En apprentissage automatique :
- **Features** (X) : Les variables d'entrée utilisées pour faire des prédictions. Ce sont les caractéristiques ou attributs de vos données.
- **Target** (y) : La variable de sortie que vous voulez prédire. C'est ce que vous essayez d'apprendre à partir des features.

## Régression Linéaire

### Concept
La régression linéaire est un algorithme d'apprentissage supervisé utilisé pour les tâches de **régression** - prédire des valeurs numériques continues.

L'algorithme trouve la droite qui s'ajuste le mieux à vos points de données. La droite est définie par :
- **Pente (coefficients)** : Combien la target change pour chaque changement unitaire d'une feature
- **Ordonnée à l'origine (intercept)** : La valeur de la target quand toutes les features sont à zéro

### Formule Mathématique
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
- y : target prédite
- β₀ : ordonnée à l'origine
- β₁, β₂, ..., βₙ : coefficients pour chaque feature
- x₁, x₂, ..., xₙ : valeurs des features
- ε : terme d'erreur

### Quand l'utiliser
- Prédire les prix des maisons en fonction de la taille, localisation, etc.
- Prévoir les ventes en fonction des dépenses publicitaires
- Estimer la température en fonction de divers facteurs météorologiques

### Exemple
```python
from sklearn.linear_model import LinearRegression

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions
predictions = model.predict(X_test)

# Vérifier les paramètres du modèle
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

## Régression Logistique

### Concept
La régression logistique est un algorithme d'apprentissage supervisé utilisé pour les tâches de **classification** - prédire des catégories ou classes discrètes.

Malgré son nom, elle est utilisée pour la classification, pas la régression. Elle prédit la probabilité qu'une instance appartienne à une classe particulière.

L'algorithme utilise la fonction logistique (sigmoïde) pour transformer la combinaison linéaire des features en une probabilité entre 0 et 1.

### Formule Mathématique
```
P(y=1|X) = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
```
- P(y=1|X) : probabilité que la target soit la classe 1 étant donné les features
- La fonction sigmoïde comprime la sortie pour être entre 0 et 1

### Quand l'utiliser
- Détection de spam dans les emails (spam/pas spam)
- Prédiction du départ des clients (partira/restera)
- Diagnostic médical (maladie/pas de maladie)
- Problèmes de classification binaire

### Exemple
```python
from sklearn.linear_model import LogisticRegression

# Créer et entraîner le modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Faire des prédictions
predictions = model.predict(X_test)  # Prédictions de classe (0 ou 1)
probabilities = model.predict_proba(X_test)  # Prédictions de probabilité
```

## Différences Clés : Régression vs Classification

| Aspect | Régression Linéaire | Régression Logistique |
|--------|-------------------|---------------------|
| **Type de Tâche** | Régression | Classification |
| **Sortie** | Valeurs continues | Classes discrètes/probabilités |
| **Exemple** | Prix maison : 250 000€ | Spam : Oui/Non |
| **Fonction de Perte** | Erreur Quadratique Moyenne | Log Loss |
| **Cas d'Usage** | Prédire des quantités | Prédire des catégories |

## Implémentation Scikit-Learn

Les deux modèles suivent le même pattern d'API dans scikit-learn :

1. **Importer** la classe du modèle
2. **Instancier** le modèle avec des paramètres
3. **Ajuster** le modèle aux données d'entraînement : `model.fit(X_train, y_train)`
4. **Prédire** sur de nouvelles données : `model.predict(X_test)`

### Paramètres Courants
- `random_state` : Pour des résultats reproductibles
- `fit_intercept` : Inclure ou non un terme d'ordonnée à l'origine (généralement True)

## Métriques d'Évaluation

### Pour la Régression (Linéaire)
- **Erreur Quadratique Moyenne (MSE)** : Moyenne des différences au carré entre prédictions et valeurs réelles
- **Score R²** : Proportion de variance expliquée par le modèle (0-1, plus élevé est mieux)

### Pour la Classification (Logistique)
- **Précision** : Fraction des prédictions correctes
- **Precision** : Vrais positifs / (Vrais positifs + Faux positifs)
- **Rappel** : Vrais positifs / (Vrais positifs + Faux négatifs)
- **Score F1** : Moyenne harmonique de la précision et du rappel

## Prochaines Étapes

Ces modèles simples forment la base pour des algorithmes plus complexes. Comprendre la régression linéaire et logistique vous aidera à :
- Interpréter les coefficients du modèle
- Comprendre l'importance des features
- Déboguer des modèles plus complexes
- Appliquer des techniques de régularisation

Dans les prochains jours, nous apprendrons l'évaluation des modèles, les splits train/test, et la gestion du surapprentissage !