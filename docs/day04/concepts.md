# Day 4: Hyperparameter Tuning & Model Selection

## What are Hyperparameters?

**Hyperparameters** are configuration variables that control the learning process of a machine learning model. Unlike model parameters (which are learned from data), hyperparameters are set before training begins.

### Examples of Hyperparameters

| Model | Hyperparameters | Description |
|-------|----------------|-------------|
| Logistic Regression | `C`, `penalty` | Regularization strength and type |
| Random Forest | `n_estimators`, `max_depth` | Number of trees and their depth |
| Ridge/Lasso | `alpha` | Regularization strength |
| Neural Networks | `learning_rate`, `batch_size` | Training configuration |

### Parameters vs Hyperparameters

- **Parameters**: Learned from data during training (e.g., `coef_`, `intercept_`)
- **Hyperparameters**: Set by you before training (e.g., `max_depth`, `C`)

## Why Tune Hyperparameters?

Default hyperparameters rarely give the best performance. Tuning helps:

1. **Improve Model Performance**: Find the optimal configuration for your specific dataset
2. **Prevent Overfitting**: Control model complexity (e.g., limit tree depth)
3. **Prevent Underfitting**: Allow enough complexity to capture patterns
4. **Optimize Training Time**: Balance accuracy with computational cost

## Grid Search Cross-Validation

### Concept

**Grid Search** exhaustively tries every combination of hyperparameters from a predefined grid.

### How It Works

1. Define a grid of hyperparameter values to test
2. For each combination:
   - Train the model using cross-validation
   - Calculate average performance
3. Return the combination with the best score

### Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}

# Create model
model = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Pros & Cons

**Pros:**
- Guaranteed to find the best combination in the grid
- Systematic and thorough

**Cons:**
- Computationally expensive (exponential growth with more parameters)
- Limited to discrete values you specify

## Random Search Cross-Validation

### Concept

**Random Search** randomly samples hyperparameter combinations from specified distributions.

### How It Works

1. Define distributions for hyperparameters (lists or statistical distributions)
2. Randomly sample `n_iter` combinations
3. Evaluate each using cross-validation
4. Return the best found combination

### Example

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),  # Random integer between 50-300
    'max_depth': [3, 5, 7, 10, None]
}

# Perform random search
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=5, random_state=42
)
random_search.fit(X, y)
```

### When to Use Random Search

- Large hyperparameter search spaces
- Limited computational budget
- Continuous hyperparameters (e.g., learning rates)

### Research Insight

[Bergstra & Bengio (2012)](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) showed that random search is often more efficient than grid search, especially when only a few hyperparameters significantly affect performance.

## Model Comparison

### Why Compare Models?

Different algorithms make different assumptions about data. Comparing multiple models helps:
- Identify which algorithm works best for your problem
- Understand trade-offs between accuracy and interpretability
- Build ensemble models using top performers

### Comparison Workflow

```python
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
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

## ML Pipelines

### The Problem: Data Leakage

When preprocessing data (scaling, imputing) before splitting:
- Information from test set leaks into training
- Leads to overly optimistic performance estimates

### The Solution: Pipelines

**Pipelines** ensure preprocessing steps are fit only on training data during cross-validation.

### Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Step 1: Scale features
    ('classifier', LogisticRegression())    # Step 2: Train model
])

# Use in grid search
param_grid = {
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)
```

### Benefits of Pipelines

1. **Prevent Data Leakage**: Preprocessing fit only on training folds
2. **Cleaner Code**: Single object for all steps
3. **Easier Deployment**: Serialize one object instead of multiple
4. **Consistent API**: Same `fit()`, `predict()` interface

## Best Practices

### 1. Start Simple
Begin with a small grid and expand based on results

### 2. Use Cross-Validation
Always use CV when tuning to avoid overfitting to a specific validation set

### 3. Choose Appropriate Scoring
- Classification: `accuracy`, `f1`, `roc_auc`, `precision`, `recall`
- Regression: `r2`, `neg_mean_squared_error`, `neg_mean_absolute_error`

### 4. Monitor Training vs Test Gap
Large gaps indicate overfitting - simplify the model

### 5. Log Your Experiments
Track hyperparameters and results for reproducibility

### 6. Consider Computation Time
- Grid Search: Tests all combinations (slow but thorough)
- Random Search: Tests fixed number (faster, often as good)

## Common Hyperparameter Grids

### Logistic Regression
```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

### Random Forest
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### Ridge Regression
```python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
```

## Next Steps

Tomorrow we'll apply everything learned to build a complete ML project - the Titanic Survival Prediction challenge. This will integrate:
- Data loading and preprocessing
- Model training with hyperparameter tuning
- Proper evaluation with cross-validation
- Model comparison and selection
