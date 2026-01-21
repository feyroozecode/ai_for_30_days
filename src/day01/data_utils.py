"""
Basic data utilities for Day 1 of AI Bootcamp
"""

import pandas as pd
import numpy as np


def load_sample_data():
    """
    Create a simple sample DataFrame for testing.

    Returns:
        pd.DataFrame: A sample dataset
    """
    data = {
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'target': np.random.randint(0, 2, 10)
    }
    return pd.DataFrame(data)


def basic_stats(df):
    """
    Calculate basic statistics for a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Basic statistics
    """
    return {
        'mean': df.select_dtypes(include=[np.number]).mean().to_dict(),
        'std': df.select_dtypes(include=[np.number]).std().to_dict(),
        'shape': df.shape
    }


if __name__ == "__main__":
    df = load_sample_data()
    print(basic_stats(df))
