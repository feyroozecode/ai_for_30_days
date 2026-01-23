"""
Tests for models.py
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.day02.models import (
    create_linear_regression_data,
    train_linear_regression,
    predict_linear_regression,
    create_logistic_regression_data,
    train_logistic_regression,
    predict_logistic_regression,
    predict_proba_logistic_regression
)


def test_create_linear_regression_data():
    """Test that create_linear_regression_data returns correct shapes"""
    X, y = create_linear_regression_data(n_samples=50, n_features=2)

    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_train_linear_regression():
    """Test that train_linear_regression returns a trained model"""
    X, y = create_linear_regression_data(n_samples=100, n_features=1)
    model = train_linear_regression(X, y)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')


def test_predict_linear_regression():
    """Test that predict_linear_regression returns predictions"""
    X, y = create_linear_regression_data(n_samples=100, n_features=1)
    model = train_linear_regression(X, y)
    predictions = predict_linear_regression(model, X[:10])

    assert predictions.shape == (10,)
    assert isinstance(predictions, np.ndarray)


def test_create_logistic_regression_data():
    """Test that create_logistic_regression_data returns correct shapes"""
    X, y = create_logistic_regression_data(n_samples=50, n_features=2, n_classes=2)

    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert set(np.unique(y)) == {0, 1}  # Binary classification


def test_train_logistic_regression():
    """Test that train_logistic_regression returns a trained model"""
    X, y = create_logistic_regression_data(n_samples=100, n_features=2)
    model = train_logistic_regression(X, y)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')


def test_predict_logistic_regression():
    """Test that predict_logistic_regression returns predictions"""
    X, y = create_logistic_regression_data(n_samples=100, n_features=2)
    model = train_logistic_regression(X, y)
    predictions = predict_logistic_regression(model, X[:10])

    assert predictions.shape == (10,)
    assert isinstance(predictions, np.ndarray)
    assert all(pred in [0, 1] for pred in predictions)


def test_predict_proba_logistic_regression():
    """Test that predict_proba_logistic_regression returns probabilities"""
    X, y = create_logistic_regression_data(n_samples=100, n_features=2)
    model = train_logistic_regression(X, y)
    probabilities = predict_proba_logistic_regression(model, X[:10])

    assert probabilities.shape == (10, 2)  # Binary classification
    assert isinstance(probabilities, np.ndarray)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1