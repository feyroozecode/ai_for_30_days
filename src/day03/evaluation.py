"""
Model evaluation and validation utilities for Day 3 of AI Bootcamp

This module covers:
- Train/test splitting
- Cross-validation
- Evaluation metrics for regression and classification
- Understanding overfitting and underfitting
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict, Any


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
    """
    Perform k-fold cross-validation on a model.
    
    Args:
        model (Any): Scikit-learn compatible model
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        cv (int): Number of folds (default: 5)
        scoring (str): Scoring metric (default: 'accuracy')
        
    Returns:
        Dict: Cross-validation results with mean and std
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        
    Returns:
        Dict: Dictionary containing MSE, MAE, RMSE, and R²
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate classification evaluation metrics.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
        
    Returns:
        Dict: Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
    }


def detect_overfitting(train_score: float, test_score: float, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect potential overfitting based on performance difference.
    
    Args:
        train_score (float): Model score on training data
        test_score (float): Model score on test data
        threshold (float): Difference threshold to consider overfitting
        
    Returns:
        Dict: Analysis results including overfitting detection
    """
    diff = abs(train_score - test_score)
    is_overfitting = diff > threshold and train_score > test_score
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'difference': diff,
        'is_overfitting': is_overfitting,
        'severity': 'high' if diff > threshold * 2 else 'moderate' if diff > threshold else 'low'
    }


def calculate_metrics_summary(regression_results: Dict[str, float] = None,
                            classification_results: Dict[str, Any] = None) -> str:
    """
    Generate a human-readable summary of evaluation metrics.
    
    Args:
        regression_results (Dict): Results from evaluate_regression
        classification_results (Dict): Results from evaluate_classification
        
    Returns:
        str: Formatted summary string
    """
    summary = []
    
    if regression_results:
        summary.append("=== REGRESSION METRICS ===")
        summary.append(f"MSE (Mean Squared Error): {regression_results['mse']:.4f}")
        summary.append(f"MAE (Mean Absolute Error): {regression_results['mae']:.4f}")
        summary.append(f"RMSE (Root Mean Squared Error): {regression_results['rmse']:.4f}")
        summary.append(f"R² Score: {regression_results['r2']:.4f}")
        summary.append("")
        
        # Interpretation
        if regression_results['r2'] > 0.8:
            summary.append("Interpretation: Excellent model performance!")
        elif regression_results['r2'] > 0.6:
            summary.append("Interpretation: Good model performance.")
        elif regression_results['r2'] > 0.4:
            summary.append("Interpretation: Moderate model performance.")
        else:
            summary.append("Interpretation: Poor model performance - consider improvements.")
            
    if classification_results:
        summary.append("=== CLASSIFICATION METRICS ===")
        summary.append(f"Accuracy: {classification_results['accuracy']:.4f}")
        summary.append(f"Precision: {classification_results['precision']:.4f}")
        summary.append(f"Recall: {classification_results['recall']:.4f}")
        summary.append(f"F1-Score: {classification_results['f1_score']:.4f}")
        summary.append("")
        
        # Interpretation
        if classification_results['accuracy'] > 0.9:
            summary.append("Interpretation: Excellent classification accuracy!")
        elif classification_results['accuracy'] > 0.8:
            summary.append("Interpretation: Good classification accuracy.")
        elif classification_results['accuracy'] > 0.7:
            summary.append("Interpretation: Fair classification accuracy.")
        else:
            summary.append("Interpretation: Poor classification accuracy - consider improvements.")
            
        summary.append("\nConfusion Matrix:")
        cm = classification_results['confusion_matrix']
        for i, row in enumerate(cm):
            summary.append(f"Class {i}: {row}")
    
    return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.datasets import make_regression, make_classification
    
    print("=== REGRESSION EXAMPLE ===")
    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X_reg, y_reg)
    
    model_reg = LinearRegression()
    model_reg.fit(X_train_reg, y_train_reg)
    
    train_pred_reg = model_reg.predict(X_train_reg)
    test_pred_reg = model_reg.predict(X_test_reg)
    
    train_metrics = evaluate_regression(y_train_reg, train_pred_reg)
    test_metrics = evaluate_regression(y_test_reg, test_pred_reg)
    
    print("Training Metrics:")
    print(calculate_metrics_summary(regression_results=train_metrics))
    print("\nTest Metrics:")
    print(calculate_metrics_summary(regression_results=test_metrics))
    
    overfit_check = detect_overfitting(train_metrics['r2'], test_metrics['r2'])
    print(f"\nOverfitting Check: {overfit_check}")
    
    print("\n" + "="*50 + "\n")
    
    print("=== CLASSIFICATION EXAMPLE ===")
    # Classification example
    X_clf, y_clf = make_classification(n_samples=1000, n_features=5, n_classes=3, 
                                     n_informative=3, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(X_clf, y_clf)
    
    model_clf = LogisticRegression(random_state=42, max_iter=1000)
    model_clf.fit(X_train_clf, y_train_clf)
    
    train_pred_clf = model_clf.predict(X_train_clf)
    test_pred_clf = model_clf.predict(X_test_clf)
    
    train_metrics_clf = evaluate_classification(y_train_clf, train_pred_clf)
    test_metrics_clf = evaluate_classification(y_test_clf, test_pred_clf)
    
    print("Training Metrics:")
    print(calculate_metrics_summary(classification_results=train_metrics_clf))
    print("\nTest Metrics:")
    print(calculate_metrics_summary(classification_results=test_metrics_clf))
    
    overfit_check_clf = detect_overfitting(train_metrics_clf['accuracy'], test_metrics_clf['accuracy'])
    print(f"\nOverfitting Check: {overfit_check_clf}")
