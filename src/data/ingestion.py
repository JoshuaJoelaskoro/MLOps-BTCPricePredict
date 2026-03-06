"""
Data ingestion module — fetches daily BTC OHLCV data from CoinGecko API.
Supports incremental ingestion based on last stored timestamp.
"""

import requests
import pandas as pd
from pathlib import Path


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


def fetch_btc_market_chart(days: int = 365, vs_currency: str = "usd") -> dict:
    """Fetch BTC historical market chart data from CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_market_chart(data: dict) -> pd.DataFrame:
    """Parse raw CoinGecko market chart response into a clean DataFrame."""
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price_usd"])
    market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap_usd"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "total_volume_usd"])

    df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df = df.sort_index()
    return df


def load_existing_data(filepath: str) -> pd.DataFrame | None:
    """Load existing raw CSV data if available."""
    path = Path(filepath)
    if path.exists():
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def incremental_fetch(output_path: str = "data/raw/btc_daily.csv",
                      days: int = 365) -> pd.DataFrame:
    """
    Fetch new BTC data incrementally. Only appends rows newer than the
    last stored timestamp, avoiding duplicates.
    """
    existing = load_existing_data(output_path)
    raw = fetch_btc_market_chart(days=days)
    new_df = parse_market_chart(raw)

    if existing is not None:
        last_ts = existing.index.max()
        new_df = new_df[new_df.index > last_ts]
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path)
    print(f"[fetch] Saved {len(combined)} rows to {output_path}. New rows: {len(new_df)}")
    return combined


if __name__ == "__main__":
    df = incremental_fetch()
    print(df.tail())