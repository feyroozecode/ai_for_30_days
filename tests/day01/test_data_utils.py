""" 
Tests for data_utils.py
"""

import pytest
import pandas as pd
import numpy as np
from src.day01.data_utils import load_sample_data, basic_stats

def test_load_sample_data():
    """ Test that load_sample_data returns a DataFrane with expected columns """
    df = load_sample_data();

    assert isinstance(df, pd.DataFrame)
    assert 'feature1' in df.columns
    assert 'feature2' in df.columns
    assert 'target' in df.columns
    assert len(df) == 10


def test_basic_stats():
    """ Test that basic_stats returns correct statistics """
    df = load_sample_data()
    stats = basic_stats(df)

    assert 'mean' in stats
    assert 'std' in stats
    assert 'shape' in stats
    assert stats['shape'] == (10, 3)
    assert isinstance(stats['mean'], dict)
    assert isinstance(stats['std'], dict)

def test_basic_stats_with_empty_df():
    """ Test that basic_stats returns correct statistics for empty data"""
    df = pd.DataFrame()
    stats = basic_stats(df)

    assert stats['shape'] == (0, 0)
    assert stats['mean'] == {}
    assert stats['std'] == {}


