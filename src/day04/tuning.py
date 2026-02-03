"""
Hyperparameter tuning and model selection for Day 04 of AI Bootcamp

This module covers:
- Grid Search Cross-Validation
- Random Search Cross-Validation
- Model comparison and selection
- Building ML pipelines with preprocessing
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score, r2_score, mean_squared_error
import warnings


def grid_search_cv(
    model: Any,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Perform Grid Search Cross-Validation to find the best hyperparameters.
    
    Grid Search exhaustively tries every combination of hyperparameters in the
    param_grid, evaluating each using cross-validation.
    
    Args:
        model: Scikit-learn estimator (unfitted)
        param_grid: Dictionary mapping parameter names to lists of values to try
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        cv (int): Number of cross-validation folds (default: 5)
        scoring (str): Scoring metric (default: None, uses model's default)
        n_jobs (int): Number of parallel jobs (-1 for all processors)
        
    Returns:
        Dict: Results containing best parameters, best score, and CV results
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> param_grid = {
        ...     'n_estimators': [50, 100, 200],
        ...     'max_depth': [3, 5, 7, None]
        ... }
        >>> results = grid_search_cv(model, param_grid, X, y, cv=5)
        >>> print(f"Best score: {results['best_score']:.4f}")
        >>> print(f"Best params: {results['best_params']}")
    """
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
                'params': grid_search.cv_results_['params']
            },
            'n_combinations_tested': len(grid_search.cv_results_['params'])
        }


def random_search_cv(
    model: Any,
    param_distributions: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 10,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Random Search Cross-Validation to find good hyperparameters.
    
    Random Search samples hyperparameter combinations randomly from the specified
    distributions. Often more efficient than Grid Search when the search space
    is large, as it doesn't exhaustively try all combinations.
    
    Args:
        model: Scikit-learn estimator (unfitted)
        param_distributions: Dictionary mapping parameter names to distributions
                            (lists for discrete values, scipy distributions for continuous)
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        n_iter (int): Number of parameter settings sampled (default: 10)
        cv (int): Number of cross-validation folds (default: 5)
        scoring (str): Scoring metric (default: None, uses model's default)
        n_jobs (int): Number of parallel jobs (-1 for all processors)
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict: Results containing best parameters, best score, and CV results
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from scipy.stats import randint
        >>> model = RandomForestClassifier(random_state=42)
        >>> param_dist = {
        ...     'n_estimators': randint(50, 300),
        ...     'max_depth': [3, 5, 7, 10, None]
        ... }
        >>> results = random_search_cv(model, param_dist, X, y, n_iter=20)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'cv_results': {
                'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': random_search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': random_search.cv_results_['mean_train_score'].tolist(),
                'params': random_search.cv_results_['params']
            },
            'n_combinations_tested': len(random_search.cv_results_['params'])
        }


def compare_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare multiple models using cross-validation.
    
    Args:
        models (Dict[str, Any]): Dictionary mapping model names to unfitted estimators
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        cv (int): Number of cross-validation folds (default: 5)
        scoring (str): Scoring metric (default: None, uses model's default)
        
    Returns:
        Dict: Comparison results for all models
        
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> models = {
        ...     'Logistic Regression': LogisticRegression(max_iter=1000),
        ...     'Random Forest': RandomForestClassifier(n_estimators=100)
        ... }
        >>> results = compare_models(models, X, y, cv=5)
        >>> best_model_name = results['best_model_name']
    """
    from sklearn.model_selection import cross_val_score
    
    results = {}
    model_scores = {}
    
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }
        model_scores[name] = scores.mean()
    
    # Find best model
    best_model_name = max(model_scores, key=model_scores.get)
    
    return {
        'model_results': results,
        'best_model_name': best_model_name,
        'best_score': model_scores[best_model_name],
        'ranking': sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    }


def create_pipeline(
    steps: List[Tuple[str, Any]],
    scaler_type: Optional[str] = 'standard'
) -> Pipeline:
    """
    Create a scikit-learn pipeline with preprocessing and a model.
    
    Pipelines ensure that preprocessing steps (like scaling) are fit only on
    training data during cross-validation, preventing data leakage.
    
    Args:
        steps (List[Tuple[str, Any]]): List of (name, transformer/estimator) tuples
        scaler_type (str): Type of scaler to add ('standard', 'minmax', or None)
        
    Returns:
        Pipeline: Configured scikit-learn pipeline
        
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = create_pipeline(
        ...     [('classifier', LogisticRegression(max_iter=1000))],
        ...     scaler_type='standard'
        ... )
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    pipeline_steps = []
    
    # Add scaler if specified
    if scaler_type == 'standard':
        pipeline_steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'minmax':
        pipeline_steps.append(('scaler', MinMaxScaler()))
    
    # Add remaining steps
    pipeline_steps.extend(steps)
    
    return Pipeline(pipeline_steps)


def get_best_model(
    comparison_results: Dict[str, Any],
    models: Dict[str, Any]
) -> Any:
    """
    Get the best performing model from comparison results.
    
    Args:
        comparison_results (Dict[str, Any]): Results from compare_models()
        models (Dict[str, Any]): Original models dictionary
        
    Returns:
        Any: The best performing model instance
    """
    best_name = comparison_results['best_model_name']
    return models[best_name]


def suggest_hyperparameter_grids(model_name: str) -> Dict[str, List[Any]]:
    """
    Suggest hyperparameter grids for common models.
    
    Args:
        model_name (str): Name of the model ('logistic_regression', 
                         'random_forest_classifier', 'ridge', 'lasso', etc.)
                         
    Returns:
        Dict[str, List[Any]]: Suggested parameter grid for GridSearchCV
        
    Example:
        >>> param_grid = suggest_hyperparameter_grids('random_forest_classifier')
        >>> print(param_grid)
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None], ...}
    """
    grids = {
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        },
        'random_forest_classifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'random_forest_regressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        },
        'lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'max_iter': [1000, 2000, 5000],
            'selection': ['cyclic', 'random']
        }
    }
    
    return grids.get(model_name.lower(), {})


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    
    print("=" * 60)
    print("DAY 04: HYPERPARAMETER TUNING & MODEL SELECTION")
    print("=" * 60)
    
    # Classification Example
    print("\n--- Classification Example ---")
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=10, n_classes=2, 
        n_informative=5, random_state=42
    )
    
    # Compare multiple models
    print("\n1. Comparing Multiple Models:")
    models_to_compare = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    comparison = compare_models(models_to_compare, X_clf, y_clf, cv=5)
    print(f"Best Model: {comparison['best_model_name']}")
    print(f"Best Score: {comparison['best_score']:.4f}")
    print("\nAll Results:")
    for name, result in comparison['model_results'].items():
        print(f"  {name}: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})")
    
    # Grid Search Example
    print("\n2. Grid Search CV for Random Forest:")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_results = grid_search_cv(rf, param_grid, X_clf, y_clf, cv=3)
    print(f"Best Score: {grid_results['best_score']:.4f}")
    print(f"Best Params: {grid_results['best_params']}")
    print(f"Combinations Tested: {grid_results['n_combinations_tested']}")
    
    # Random Search Example
    print("\n3. Random Search CV for Random Forest:")
    from scipy.stats import randint
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [3, 5, 7, 10, None]
    }
    random_results = random_search_cv(
        rf, param_dist, X_clf, y_clf, n_iter=5, cv=3, random_state=42
    )
    print(f"Best Score: {random_results['best_score']:.4f}")
    print(f"Best Params: {random_results['best_params']}")
    
    # Pipeline Example
    print("\n4. Pipeline with Scaling:")
    pipeline = create_pipeline(
        [('classifier', LogisticRegression(max_iter=1000, random_state=42))],
        scaler_type='standard'
    )
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline, X_clf, y_clf, cv=5)
    print(f"Pipeline CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Regression Example
    print("\n--- Regression Example ---")
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=10, noise=0.1, random_state=42
    )
    
    print("\n5. Comparing Regression Models:")
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    reg_comparison = compare_models(reg_models, X_reg, y_reg, cv=5, scoring='r2')
    print(f"Best Model: {reg_comparison['best_model_name']}")
    print(f"Best R² Score: {reg_comparison['best_score']:.4f}")
    print("\nAll Results:")
    for name, result in reg_comparison['model_results'].items():
        print(f"  {name}: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})")
    
    # Hyperparameter grid suggestion
    print("\n6. Suggested Hyperparameter Grids:")
    print("Logistic Regression:")
    print(suggest_hyperparameter_grids('logistic_regression'))
    print("\nRandom Forest Classifier:")
    print(suggest_hyperparameter_grids('random_forest_classifier'))
