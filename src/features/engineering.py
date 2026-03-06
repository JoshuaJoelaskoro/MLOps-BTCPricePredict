"""
Feature engineering module — computes technical indicators and
the binary classification target (price direction: 1=up, 0=down).
"""

import pandas as pd
import numpy as np


def compute_returns(df: pd.DataFrame, windows: list = [1, 3, 7, 14]) -> pd.DataFrame:
    """Calculate rolling daily returns for given lookback windows."""
    for w in windows:
        df[f"return_{w}d"] = df["price_usd"].pct_change(w)
    return df


def compute_moving_averages(df: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Calculate simple moving averages and price-to-MA ratio."""
    for w in windows:
        df[f"ma_{w}d"] = df["price_usd"].rolling(w).mean()
        df[f"price_to_ma_{w}d"] = df["price_usd"] / df[f"ma_{w}d"]
    return df


def compute_volatility(df: pd.DataFrame, windows: list = [7, 14]) -> pd.DataFrame:
    """Calculate rolling volatility (std of daily returns)."""
    daily_return = df["price_usd"].pct_change(1)
    for w in windows:
        df[f"volatility_{w}d"] = daily_return.rolling(w).std()
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    df["volume_ma_7d"] = df["total_volume_usd"].rolling(7).mean()
    df["volume_ratio"] = df["total_volume_usd"] / df["volume_ma_7d"]
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if next-day price is higher than today, else 0.
    Uses next-day close (forward shift by 1).
    """
    df["next_price"] = df["price_usd"].shift(-1)
    df["direction"] = (df["next_price"] > df["price_usd"]).astype(int)
    df.drop(columns=["next_price"], inplace=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = compute_returns(df)
    df = compute_moving_averages(df)
    df = compute_volatility(df)
    df = compute_volume_features(df)
    df = create_target(df)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "price_usd": [40000, 41000, 40500, 42000, 41500, 43000, 44000,
                      43500, 45000, 46000, 45500, 47000, 48000, 47500,
                      49000, 50000, 49500, 51000, 52000, 51500, 53000,
                      54000, 53500, 55000, 56000, 55500, 57000, 58000,
                      57500, 59000, 60000],
        "total_volume_usd": [1e10] * 31,
        "market_cap_usd": [8e11] * 31,
    }, index=pd.date_range("2024-01-01", periods=31, freq="D", tz="UTC"))

    features = build_features(sample)
    print(features.tail())
    print("Features:", list(features.columns))