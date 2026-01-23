"""
Basic machine learning models for Day 2 of AI Bootcamp
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification


def create_linear_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=42):
    """
    Create sample data for linear regression.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise (float): Amount of noise
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                          noise=noise, random_state=random_state)
    return X, y


def train_linear_regression(X, y):
    """
    Train a linear regression model.

    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector

    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_linear_regression(model, X):
    """
    Make predictions with a linear regression model.

    Args:
        model (LinearRegression): Trained model
        X (array-like): Feature matrix

    Returns:
        array: Predictions
    """
    return model.predict(X)


def create_logistic_regression_data(n_samples=100, n_features=2, n_classes=2, random_state=42):
    """
    Create sample data for logistic regression.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        n_classes (int): Number of classes (for binary, use 2)
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                              n_classes=n_classes, n_redundant=0, n_informative=n_features,
                              random_state=random_state, n_clusters_per_class=1)
    return X, y


def train_logistic_regression(X, y):
    """
    Train a logistic regression model.

    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector

    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


def predict_logistic_regression(model, X):
    """
    Make predictions with a logistic regression model.

    Args:
        model (LogisticRegression): Trained model
        X (array-like): Feature matrix

    Returns:
        array: Predictions (class labels)
    """
    return model.predict(X)


def predict_proba_logistic_regression(model, X):
    """
    Make probability predictions with a logistic regression model.

    Args:
        model (LogisticRegression): Trained model
        X (array-like): Feature matrix

    Returns:
        array: Prediction probabilities
    """
    return model.predict_proba(X)


if __name__ == "__main__":
    # Linear Regression Example
    print("Linear Regression Example:")
    X_lr, y_lr = create_linear_regression_data()
    model_lr = train_linear_regression(X_lr, y_lr)
    predictions_lr = predict_linear_regression(model_lr, X_lr[:5])
    print(f"Predictions: {predictions_lr}")
    print(f"Coefficients: {model_lr.coef_}")
    print(f"Intercept: {model_lr.intercept_}")

    print("\nLogistic Regression Example:")
    X_log, y_log = create_logistic_regression_data()
    model_log = train_logistic_regression(X_log, y_log)
    predictions_log = predict_logistic_regression(model_log, X_log[:5])
    probabilities = predict_proba_logistic_regression(model_log, X_log[:5])
    print(f"Predictions: {predictions_log}")
    print(f"Probabilities: {probabilities}")