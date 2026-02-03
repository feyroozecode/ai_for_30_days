"""
Day 04: Hyperparameter Tuning & Model Selection

This module covers:
- Hyperparameter tuning with Grid Search and Random Search
- Model comparison and selection
- Building complete ML pipelines
"""

from .tuning import (
    grid_search_cv,
    random_search_cv,
    compare_models,
    create_pipeline,
    get_best_model,
    suggest_hyperparameter_grids
)

__all__ = [
    'grid_search_cv',
    'random_search_cv',
    'compare_models',
    'create_pipeline',
    'get_best_model',
    'suggest_hyperparameter_grids'
]
