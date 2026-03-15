"""
Data collector for NASDAQ stocks.
Downloads OHLCV data from Yahoo Finance for NASDAQ 100 components.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not installed. Install with: pip install yfinance")


def download_stock_data(
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "~/.qlib/csv_data/nasdaq",
    retry: int = 3,
    pause: float = 0.5,
) -> dict:
    """
    Download OHLCV data for a list of symbols from Yahoo Finance.

    Args:
        symbols: list of ticker symbols
        start_date: start date string
        end_date: end date string
        output_dir: directory to save CSV files
        retry: number of retries per symbol
        pause: pause between requests in seconds

    Returns:
        dict with {symbol: filepath} for successful downloads
    """
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    failed = []

    print(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")

    for i, symbol in enumerate(symbols):
        filepath = os.path.join(output_dir, f"{symbol}.csv")

        # Skip if already downloaded
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 100:
                results[symbol] = filepath
                print(f"  [{i+1}/{len(symbols)}] {symbol}: already exists ({len(df)} rows)")
                continue

        success = False
        for attempt in range(1, retry + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

                if df is None or len(df) < 10:
                    print(f"  [{i+1}/{len(symbols)}] {symbol}: insufficient data (attempt {attempt})")
                    time.sleep(pause * 2)
                    continue

                # Standardize columns
                df = df.reset_index()
                df = df.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })

                # Handle timezone-aware datetimes
                if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                # Keep only needed columns
                df = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Remove rows with zero/nan values
                df = df.replace(0, np.nan).dropna()

                if len(df) < 10:
                    continue

                df.to_csv(filepath, index=False)
                results[symbol] = filepath
                success = True
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} rows downloaded")
                break

            except Exception as e:
                print(f"  [{i+1}/{len(symbols)}] {symbol}: error (attempt {attempt}): {e}")
                time.sleep(pause * attempt)

        if not success:
            failed.append(symbol)
            print(f"  [{i+1}/{len(symbols)}] {symbol}: FAILED after {retry} attempts")

        time.sleep(pause)

    print(f"\nDownload complete: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed symbols: {failed}")

    return results


def download_benchmark(
    symbol: str = "^NDX",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "~/.qlib/csv_data/nasdaq",
) -> Optional[str]:
    """Download benchmark index data."""
    if yf is None:
        raise ImportError("yfinance is required.")

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "benchmark_NDX.csv")
    if os.path.exists(filepath):
        print(f"Benchmark data already exists: {filepath}")
        return filepath

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if df is None or len(df) < 10:
            # Fallback: use QQQ ETF as NASDAQ proxy
            print(f"Could not download {symbol}, trying QQQ as proxy...")
            ticker = yf.Ticker("QQQ")
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.replace(0, np.nan).dropna()

        df.to_csv(filepath, index=False)
        print(f"Benchmark data downloaded: {len(df)} rows -> {filepath}")
        return filepath

    except Exception as e:
        print(f"Failed to download benchmark: {e}")
        return None
