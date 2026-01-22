# Jour 1 : Configuration de l'environnement

## Vue d'ensemble

Ce document vous guide à travers la configuration de votre environnement de développement pour le bootcamp IA pour 30 jours.

## Prérequis

- Python 3.8 ou supérieur installé
- Gestionnaire de paquets pip
- Git

## Étapes

### 1. Créer un environnement virtuel

```bash
python -m venv ai_bootcamp_env
```

### 2. Activer l'environnement virtuel

**Sur macOS/Linux :**
```bash
source ai_bootcamp_env/bin/activate
```

**Sur Windows :**
```bash
ai_bootcamp_env\Scripts\activate
```

### 3. Installer les bibliothèques requises

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Toutes les bibliothèques installées avec succès !")
```

### 5. Configurer la structure du projet

Le projet suit cette structure :
- `docs/` : Documentation
- `src/` : Code source
- `tests/` : Fichiers de test
- `README.md` : Vue d'ensemble du projet
- `requirements.txt` : Dépendances

## Prochaines étapes

Une fois votre environnement configuré, passez à l'apprentissage des concepts de base dans [concepts.md](concepts.md).
