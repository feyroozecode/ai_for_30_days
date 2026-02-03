"""
Tests for tuning.py module
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

from src.day04.tuning import (
    grid_search_cv,
    random_search_cv,
    compare_models,
    create_pipeline,
    get_best_model,
    suggest_hyperparameter_grids
)


# Fixtures
@pytest.fixture
def classification_data():
    """Create sample classification data"""
    X, y = make_classification(
        n_samples=200, n_features=5, n_classes=2,
        n_informative=3, random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Create sample regression data"""
    X, y = make_regression(
        n_samples=200, n_features=5, noise=0.1, random_state=42
    )
    return X, y


# Tests for grid_search_cv
def test_grid_search_cv_basic(classification_data):
    """Test basic grid search functionality"""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
    
    results = grid_search_cv(model, param_grid, X, y, cv=3)
    
    # Check results structure
    assert 'best_params' in results
    assert 'best_score' in results
    assert 'best_estimator' in results
    assert 'cv_results' in results
    assert 'n_combinations_tested' in results
    
    # Check types
    assert isinstance(results['best_params'], dict)
    assert isinstance(results['best_score'], float)
    assert isinstance(results['n_combinations_tested'], int)
    
    # Check that all combinations were tested
    assert results['n_combinations_tested'] == 3  # 3 values of C
    
    # Score should be between 0 and 1 for classification
    assert 0 <= results['best_score'] <= 1


def test_grid_search_cv_with_random_forest(classification_data):
    """Test grid search with Random Forest"""
    X, y = classification_data
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    param_grid = {
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    
    results = grid_search_cv(model, param_grid, X, y, cv=3)
    
    # Should test 3 * 2 = 6 combinations
    assert results['n_combinations_tested'] == 6
    
    # Best params should be in the grid
    assert results['best_params']['max_depth'] in [3, 5, None]
    assert results['best_params']['min_samples_split'] in [2, 5]


def test_grid_search_cv_regression(regression_data):
    """Test grid search with regression model"""
    X, y = regression_data
    model = Ridge(random_state=42)
    param_grid = {
        'alpha': [0.1, 1.0, 10.0]
    }
    
    results = grid_search_cv(model, param_grid, X, y, cv=3, scoring='r2')
    
    assert results['n_combinations_tested'] == 3
    assert 'alpha' in results['best_params']
    # R² score can be negative but should be reasonable
    assert -1 <= results['best_score'] <= 1


def test_grid_search_cv_results_structure(classification_data):
    """Test that cv_results has expected structure"""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {'C': [0.1, 1.0]}
    
    results = grid_search_cv(model, param_grid, X, y, cv=3)
    cv_results = results['cv_results']
    
    assert 'mean_test_score' in cv_results
    assert 'std_test_score' in cv_results
    assert 'mean_train_score' in cv_results
    assert 'params' in cv_results
    
    # Check lengths match number of combinations
    assert len(cv_results['mean_test_score']) == 2
    assert len(cv_results['params']) == 2


# Tests for random_search_cv
def test_random_search_cv_basic(classification_data):
    """Test basic random search functionality"""
    X, y = classification_data
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    param_dist = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    results = random_search_cv(model, param_dist, X, y, n_iter=3, cv=3, random_state=42)
    
    # Check results structure
    assert 'best_params' in results
    assert 'best_score' in results
    assert 'best_estimator' in results
    assert 'n_combinations_tested' in results
    
    # Should test exactly n_iter combinations
    assert results['n_combinations_tested'] == 3


def test_random_search_cv_with_scipy_distributions(classification_data):
    """Test random search with scipy distributions"""
    X, y = classification_data
    model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': randint(10, 50),
        'max_depth': [3, 5, 7, None]
    }
    
    results = random_search_cv(model, param_dist, X, y, n_iter=5, cv=3, random_state=42)
    
    assert results['n_combinations_tested'] == 5
    assert 10 <= results['best_params']['n_estimators'] < 50


def test_random_search_cv_regression(regression_data):
    """Test random search with regression"""
    X, y = regression_data
    model = Ridge(random_state=42)
    param_dist = {
        'alpha': uniform(0.001, 10)
    }
    
    results = random_search_cv(model, param_dist, X, y, n_iter=5, cv=3, scoring='r2', random_state=42)
    
    assert results['n_combinations_tested'] == 5
    assert 0.001 <= results['best_params']['alpha'] <= 10.001


# Tests for compare_models
def test_compare_models_basic(classification_data):
    """Test basic model comparison"""
    X, y = classification_data
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    results = compare_models(models, X, y, cv=3)
    
    # Check structure
    assert 'model_results' in results
    assert 'best_model_name' in results
    assert 'best_score' in results
    assert 'ranking' in results
    
    # Both models should have results
    assert 'Logistic Regression' in results['model_results']
    assert 'Random Forest' in results['model_results']
    
    # Best model should be one of the two
    assert results['best_model_name'] in models.keys()
    
    # Ranking should have both models
    assert len(results['ranking']) == 2


def test_compare_models_results_structure(classification_data):
    """Test that individual model results have expected structure"""
    X, y = classification_data
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = compare_models(models, X, y, cv=3)
    model_result = results['model_results']['Logistic Regression']
    
    assert 'mean_score' in model_result
    assert 'std_score' in model_result
    assert 'scores' in model_result
    assert 'min_score' in model_result
    assert 'max_score' in model_result
    
    assert isinstance(model_result['mean_score'], float)
    assert isinstance(model_result['scores'], list)
    assert len(model_result['scores']) == 3  # cv=3


def test_compare_models_with_scoring(classification_data):
    """Test model comparison with different scoring metrics"""
    X, y = classification_data
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    # Test with f1 scoring
    results_f1 = compare_models(models, X, y, cv=3, scoring='f1')
    assert 0 <= results_f1['best_score'] <= 1


def test_compare_models_regression(regression_data):
    """Test model comparison with regression models"""
    X, y = regression_data
    models = {
        'Ridge': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42)
    }
    
    results = compare_models(models, X, y, cv=3, scoring='r2')
    
    assert 'Ridge' in results['model_results']
    assert 'Random Forest' in results['model_results']
    assert results['best_model_name'] in models.keys()


# Tests for create_pipeline
def test_create_pipeline_with_standard_scaler(classification_data):
    """Test pipeline creation with standard scaler"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = create_pipeline([('classifier', model)], scaler_type='standard')
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'scaler'
    assert pipeline.steps[1][0] == 'classifier'
    
    # Should be able to fit and predict
    X, y = classification_data
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


def test_create_pipeline_with_minmax_scaler(classification_data):
    """Test pipeline creation with minmax scaler"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = create_pipeline([('classifier', model)], scaler_type='minmax')
    
    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[0][0] == 'scaler'
    
    # Verify it's MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    assert isinstance(pipeline.steps[0][1], MinMaxScaler)


def test_create_pipeline_no_scaler(classification_data):
    """Test pipeline creation without scaler"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = create_pipeline([('classifier', model)], scaler_type=None)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][0] == 'classifier'


def test_create_pipeline_multiple_steps():
    """Test pipeline with multiple steps"""
    from sklearn.decomposition import PCA
    
    steps = [
        ('pca', PCA(n_components=2)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]
    pipeline = create_pipeline(steps, scaler_type='standard')
    
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == 'scaler'
    assert pipeline.steps[1][0] == 'pca'
    assert pipeline.steps[2][0] == 'classifier'


# Tests for get_best_model
def test_get_best_model(classification_data):
    """Test getting the best model from comparison results"""
    X, y = classification_data
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    comparison = compare_models(models, X, y, cv=3)
    best_model = get_best_model(comparison, models)
    
    # Best model should match the name
    best_name = comparison['best_model_name']
    assert type(best_model) == type(models[best_name])


# Tests for suggest_hyperparameter_grids
def test_suggest_hyperparameter_grids_logistic_regression():
    """Test hyperparameter grid suggestion for logistic regression"""
    grid = suggest_hyperparameter_grids('logistic_regression')
    
    assert 'C' in grid
    assert 'penalty' in grid
    assert 'solver' in grid
    assert 'max_iter' in grid
    
    assert isinstance(grid['C'], list)
    assert len(grid['C']) > 0


def test_suggest_hyperparameter_grids_random_forest_classifier():
    """Test hyperparameter grid suggestion for random forest classifier"""
    grid = suggest_hyperparameter_grids('random_forest_classifier')
    
    assert 'n_estimators' in grid
    assert 'max_depth' in grid
    assert 'min_samples_split' in grid
    assert 'min_samples_leaf' in grid


def test_suggest_hyperparameter_grids_random_forest_regressor():
    """Test hyperparameter grid suggestion for random forest regressor"""
    grid = suggest_hyperparameter_grids('random_forest_regressor')
    
    assert 'n_estimators' in grid
    assert 'max_depth' in grid


def test_suggest_hyperparameter_grids_ridge():
    """Test hyperparameter grid suggestion for ridge regression"""
    grid = suggest_hyperparameter_grids('ridge')
    
    assert 'alpha' in grid
    assert 'solver' in grid


def test_suggest_hyperparameter_grids_lasso():
    """Test hyperparameter grid suggestion for lasso regression"""
    grid = suggest_hyperparameter_grids('lasso')
    
    assert 'alpha' in grid
    assert 'max_iter' in grid
    assert 'selection' in grid


def test_suggest_hyperparameter_grids_unknown_model():
    """Test hyperparameter grid suggestion for unknown model"""
    grid = suggest_hyperparameter_grids('unknown_model')
    
    assert grid == {}


def test_suggest_hyperparameter_grids_case_insensitive():
    """Test that model name is case insensitive"""
    grid_lower = suggest_hyperparameter_grids('logistic_regression')
    grid_upper = suggest_hyperparameter_grids('LOGISTIC_REGRESSION')
    grid_mixed = suggest_hyperparameter_grids('Logistic_Regression')
    
    assert grid_lower == grid_upper == grid_mixed


# Integration tests
def test_integration_full_workflow_classification(classification_data):
    """Integration test: full classification workflow"""
    X, y = classification_data
    
    # Step 1: Compare models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    comparison = compare_models(models, X, y, cv=3)
    
    # Step 2: Get best model and tune it
    best_model_name = comparison['best_model_name']
    best_model = models[best_model_name]
    
    if best_model_name == 'Random Forest':
        param_grid = {'max_depth': [3, 5, None]}
    else:
        param_grid = {'C': [0.1, 1, 10]}
    
    tuning_results = grid_search_cv(best_model, param_grid, X, y, cv=3)
    
    # Verify tuning improved or maintained score
    original_score = comparison['model_results'][best_model_name]['mean_score']
    tuned_score = tuning_results['best_score']
    
    # Tuned score should be close to or better than original
    assert tuned_score >= original_score - 0.1  # Allow small margin


def test_integration_full_workflow_regression(regression_data):
    """Integration test: full regression workflow"""
    X, y = regression_data
    
    # Step 1: Create pipeline
    model = Ridge(random_state=42)
    pipeline = create_pipeline([('regressor', model)], scaler_type='standard')
    
    # Step 2: Compare with and without pipeline
    models = {
        'Without Pipeline': Ridge(random_state=42),
        'With Pipeline': pipeline
    }
    comparison = compare_models(models, X, y, cv=3, scoring='r2')
    
    # Both should produce valid results
    assert 'Without Pipeline' in comparison['model_results']
    assert 'With Pipeline' in comparison['model_results']
    
    # Step 3: Tune the best approach
    best_model = Ridge(random_state=42)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    tuning_results = grid_search_cv(best_model, param_grid, X, y, cv=3, scoring='r2')
    
    assert tuning_results['best_score'] > 0  # Should have some predictive power


def test_integration_random_vs_grid_search(classification_data):
    """Compare random search and grid search on same problem"""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Grid search
    param_grid = {'max_depth': [3, 5, 7, None]}
    grid_results = grid_search_cv(model, param_grid, X, y, cv=3)
    
    # Random search
    param_dist = {'max_depth': [3, 5, 7, None]}
    random_results = random_search_cv(model, param_dist, X, y, n_iter=3, cv=3, random_state=42)
    
    # Both should find valid solutions
    assert 0 <= grid_results['best_score'] <= 1
    assert 0 <= random_results['best_score'] <= 1
    
    # Grid search should test all 4 combinations
    assert grid_results['n_combinations_tested'] == 4
    
    # Random search should test exactly n_iter combinations
    assert random_results['n_combinations_tested'] == 3
