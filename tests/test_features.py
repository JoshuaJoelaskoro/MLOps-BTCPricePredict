"""
Basic unit tests for feature engineering module.
"""

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.engineering import (
    compute_returns,
    compute_moving_averages,
    compute_volatility,
    create_target,
    build_features,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "price_usd": [40000, 41000, 40500, 42000, 41500, 43000, 44000,
                      43500, 45000, 46000, 45500, 47000, 48000, 47500,
                      49000, 50000, 49500, 51000, 52000, 51500, 53000,
                      54000, 53500, 55000, 56000, 55500, 57000, 58000,
                      57500, 59000, 60000],
        "total_volume_usd": [1e10] * 31,
        "market_cap_usd": [8e11] * 31,
    }, index=pd.date_range("2024-01-01", periods=31, freq="D", tz="UTC"))


def test_compute_returns(sample_df):
    """Test that return columns are created."""
    df = compute_returns(sample_df.copy())
    assert "return_1d" in df.columns
    assert "return_7d" in df.columns


def test_compute_moving_averages(sample_df):
    """Test that moving average columns are created."""
    df = compute_moving_averages(sample_df.copy())
    assert "ma_7d" in df.columns
    assert "price_to_ma_7d" in df.columns


def test_compute_volatility(sample_df):
    """Test that volatility columns are created."""
    df = compute_volatility(sample_df.copy())
    assert "volatility_7d" in df.columns


def test_create_target(sample_df):
    """Test that target column contains only 0 and 1."""
    df = create_target(sample_df.copy())
    assert "direction" in df.columns
    assert set(df["direction"].dropna().unique()).issubset({0, 1})


def test_build_features(sample_df):
    """Test full feature pipeline runs without error."""
    df = build_features(sample_df.copy())
    assert "direction" in df.columns
    assert len(df) > 0