"""
Tests for evaluation.py module
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from src.day03.evaluation import (
    split_data,
    cross_validate_model,
    evaluate_regression,
    evaluate_classification,
    detect_overfitting,
    calculate_metrics_summary
)


def test_split_data():
    """Test train/test split functionality"""
    # Create sample data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Check shapes
    assert X_train.shape[0] == 80  # 80% of 100
    assert X_test.shape[0] == 20   # 20% of 100
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20
    
    # Check that all data is accounted for
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    
    # Check data types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_split_data_different_sizes():
    """Test split_data with different test sizes"""
    X = np.random.rand(100, 3)
    y = np.random.rand(100)
    
    # Test 30% test size
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    assert X_train.shape[0] == 70
    assert X_test.shape[0] == 30
    
    # Test 10% test size
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.1)
    assert X_train.shape[0] == 90
    assert X_test.shape[0] == 10


def test_cross_validate_model_regression():
    """Test cross-validation for regression model"""
    # Create regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model = LinearRegression()
    
    results = cross_validate_model(model, X, y, cv=5, scoring='r2')
    
    # Check results structure
    assert 'scores' in results
    assert 'mean' in results
    assert 'std' in results
    assert 'min' in results
    assert 'max' in results
    
    # Check data types
    assert isinstance(results['scores'], np.ndarray)
    assert isinstance(results['mean'], float)
    assert isinstance(results['std'], float)
    
    # Check that we have 5 scores for 5-fold CV
    assert len(results['scores']) == 5
    
    # Scores should be reasonable for regression
    assert all(-2 <= score <= 1 for score in results['scores'])  # R² can be negative


def test_cross_validate_model_classification():
    """Test cross-validation for classification model"""
    # Create classification data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    results = cross_validate_model(model, X, y, cv=3, scoring='accuracy')
    
    # Check results structure
    assert 'scores' in results
    assert 'mean' in results
    assert 'std' in results
    assert 'min' in results
    assert 'max' in results
    
    # Check that we have 3 scores for 3-fold CV
    assert len(results['scores']) == 3
    
    # Accuracy scores should be between 0 and 1
    assert all(0 <= score <= 1 for score in results['scores'])


def test_evaluate_regression():
    """Test regression evaluation metrics"""
    # Create sample predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    
    metrics = evaluate_regression(y_true, y_pred)
    
    # Check all expected metrics are present
    expected_keys = ['mse', 'mae', 'rmse', 'r2']
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)
    
    # MSE should be positive
    assert metrics['mse'] >= 0
    
    # RMSE should equal sqrt(MSE)
    assert abs(metrics['rmse'] - np.sqrt(metrics['mse'])) < 1e-10
    
    # MAE should be positive
    assert metrics['mae'] >= 0
    
    # R² should be between -∞ and 1 (can be negative for bad models)
    assert metrics['r2'] <= 1


def test_evaluate_regression_perfect_predictions():
    """Test regression metrics with perfect predictions"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect predictions
    
    metrics = evaluate_regression(y_true, y_pred)
    
    # Perfect predictions should have:
    assert metrics['mse'] == 0.0
    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['r2'] == 1.0  # Perfect R² score


def test_evaluate_classification():
    """Test classification evaluation metrics"""
    # Create sample predictions
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    
    metrics = evaluate_classification(y_true, y_pred)
    
    # Check all expected metrics are present
    expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
    for key in expected_keys:
        assert key in metrics
    
    # Check data types
    assert isinstance(metrics['accuracy'], float)
    assert isinstance(metrics['precision'], float)
    assert isinstance(metrics['recall'], float)
    assert isinstance(metrics['f1_score'], float)
    assert isinstance(metrics['confusion_matrix'], list)
    
    # Metrics should be between 0 and 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1


def test_evaluate_classification_binary():
    """Test classification with binary classes"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    metrics = evaluate_classification(y_true, y_pred)
    
    # Confusion matrix should be 2x2 for binary classification
    cm = metrics['confusion_matrix']
    assert len(cm) == 2  # 2 classes
    assert len(cm[0]) == 2  # 2 classes
    assert len(cm[1]) == 2  # 2 classes


def test_detect_overfitting_no_overfitting():
    """Test overfitting detection when there's no overfitting"""
    result = detect_overfitting(train_score=0.85, test_score=0.82, threshold=0.1)
    
    assert 'train_score' in result
    assert 'test_score' in result
    assert 'difference' in result
    assert 'is_overfitting' in result
    assert 'severity' in result
    
    assert result['train_score'] == 0.85
    assert result['test_score'] == 0.82
    assert abs(result['difference'] - 0.03) < 1e-10
    assert result['is_overfitting'] == False  # Small difference, not overfitting
    assert result['severity'] == 'low'


def test_detect_overfitting_moderate():
    """Test overfitting detection with moderate overfitting"""
    result = detect_overfitting(train_score=0.95, test_score=0.80, threshold=0.1)
    
    assert abs(result['difference'] - 0.15) < 1e-10
    assert result['is_overfitting'] == True
    assert result['severity'] == 'moderate'


def test_detect_overfitting_severe():
    """Test overfitting detection with severe overfitting"""
    result = detect_overfitting(train_score=0.98, test_score=0.70, threshold=0.1)
    
    assert result['difference'] == 0.28
    assert result['is_overfitting'] == True
    assert result['severity'] == 'high'


def test_calculate_metrics_summary_regression():
    """Test summary generation for regression metrics"""
    regression_results = {
        'mse': 0.5,
        'mae': 0.3,
        'rmse': 0.707,
        'r2': 0.85
    }
    
    summary = calculate_metrics_summary(regression_results=regression_results)
    
    # Check that summary contains expected information
    assert 'REGRESSION METRICS' in summary
    assert 'MSE' in summary
    assert 'MAE' in summary
    assert 'RMSE' in summary
    assert 'R² Score' in summary
    assert '0.85' in summary  # R² value should appear
    assert 'Interpretation:' in summary


def test_calculate_metrics_summary_classification():
    """Test summary generation for classification metrics"""
    classification_results = {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.91,
        'f1_score': 0.90,
        'confusion_matrix': [[45, 3], [2, 50]]
    }
    
    summary = calculate_metrics_summary(classification_results=classification_results)
    
    # Check that summary contains expected information
    assert 'CLASSIFICATION METRICS' in summary
    assert 'Accuracy' in summary
    assert 'Precision' in summary
    assert 'Recall' in summary
    assert 'F1-Score' in summary
    assert '0.92' in summary  # Accuracy value should appear
    assert 'Confusion Matrix' in summary


def test_calculate_metrics_summary_both():
    """Test summary generation with both regression and classification"""
    reg_results = {'mse': 0.3, 'mae': 0.2, 'rmse': 0.548, 'r2': 0.78}
    clf_results = {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85, 'confusion_matrix': [[40, 5], [10, 45]]}
    
    summary = calculate_metrics_summary(regression_results=reg_results, classification_results=clf_results)
    
    # Should contain both sections
    assert 'REGRESSION METRICS' in summary
    assert 'CLASSIFICATION METRICS' in summary


def test_edge_cases_empty_arrays():
    """Test edge cases with empty arrays"""
    # This should not crash, but behavior depends on sklearn implementation
    X_empty = np.array([]).reshape(0, 2)
    y_empty = np.array([])
    
    # Just verify functions don't crash with empty inputs
    # Actual behavior may vary and that's okay for edge cases
    try:
        split_data(X_empty, y_empty)
    except ValueError:
        # Expected for empty arrays
        pass


def test_integration_regression_workflow():
    """Integration test: complete regression workflow"""
    # Create data
    X, y = make_regression(n_samples=200, n_features=3, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_regression(y_train, train_pred)
    test_metrics = evaluate_regression(y_test, test_pred)
    
    # Check metrics are reasonable
    assert train_metrics['r2'] > 0.8  # Should be decent on training data
    assert test_metrics['r2'] > 0.7   # Should generalize reasonably well
    
    # Check overfitting
    overfit_result = detect_overfitting(train_metrics['r2'], test_metrics['r2'])
    # Should not be severely overfitting on this simple problem
    assert overfit_result['severity'] != 'high'


def test_integration_classification_workflow():
    """Integration test: complete classification workflow"""
    # Create data
    X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_classification(y_train, train_pred)
    test_metrics = evaluate_classification(y_test, test_pred)
    
    # Check metrics are reasonable
    assert train_metrics['accuracy'] > 0.8  # Should be decent on training data
    assert test_metrics['accuracy'] > 0.7   # Should generalize reasonably well
    
    # Check overfitting
    overfit_result = detect_overfitting(train_metrics['accuracy'], test_metrics['accuracy'])
    # Should not be severely overfitting
    assert overfit_result['severity'] != 'high'
