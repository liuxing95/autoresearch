"""
Data processor: converts CSV data to Qlib-compatible format and
builds datasets with Alpha158 features.
"""

import os
import shutil
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV to Qlib binary conversion
# ---------------------------------------------------------------------------

def csv_to_qlib_format(
    csv_dir: str,
    qlib_dir: str,
    symbols: List[str],
    freq: str = "day",
) -> None:
    """
    Convert CSV files to Qlib-compatible directory structure.

    Qlib format:
        qlib_dir/
        ├── calendars/
        │   └── day.txt
        ├── instruments/
        │   └── nasdaq100.txt
        └── features/
            ├── AAPL/
            │   ├── open.day.bin  (np.float32 binary)
            │   ├── high.day.bin
            │   ├── low.day.bin
            │   ├── close.day.bin
            │   └── volume.day.bin
            └── ...
    """
    csv_dir = os.path.expanduser(csv_dir)
    qlib_dir = os.path.expanduser(qlib_dir)

    cal_dir = os.path.join(qlib_dir, "calendars")
    inst_dir = os.path.join(qlib_dir, "instruments")
    feat_dir = os.path.join(qlib_dir, "features")

    for d in [cal_dir, inst_dir, feat_dir]:
        os.makedirs(d, exist_ok=True)

    all_dates = set()
    symbol_ranges = {}
    fields = ["open", "high", "low", "close", "volume"]

    processed = 0

    for symbol in symbols:
        csv_path = os.path.join(csv_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 10:
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
        all_dates.update(dates)
        symbol_ranges[symbol] = (dates[0], dates[-1])

        # Write feature binaries
        sym_dir = os.path.join(feat_dir, symbol)
        os.makedirs(sym_dir, exist_ok=True)

        for field_name in fields:
            if field_name in df.columns:
                values = df[field_name].values.astype(np.float32)
                bin_path = os.path.join(sym_dir, f"{field_name}.{freq}.bin")
                values.tofile(bin_path)

        processed += 1

    # Write calendar
    sorted_dates = sorted(all_dates)
    cal_path = os.path.join(cal_dir, f"{freq}.txt")
    with open(cal_path, "w") as f:
        for d in sorted_dates:
            f.write(d + "\n")

    # Write instruments
    inst_path = os.path.join(inst_dir, "nasdaq100.txt")
    with open(inst_path, "w") as f:
        for symbol, (start, end) in sorted(symbol_ranges.items()):
            f.write(f"{symbol}\t{start}\t{end}\n")

    print(f"Qlib format conversion complete:")
    print(f"  Symbols processed: {processed}")
    print(f"  Calendar dates: {len(sorted_dates)}")
    print(f"  Output dir: {qlib_dir}")


# ---------------------------------------------------------------------------
# Feature & dataset building (standalone, no Qlib dependency for portability)
# ---------------------------------------------------------------------------

class Alpha158Builder:
    """
    Build Alpha158-style features from OHLCV data.
    Produces ~158 factors covering price, volume, momentum, volatility, etc.
    """

    WINDOWS = [5, 10, 20, 30, 60]

    def __init__(self, csv_dir: str, symbols: List[str]):
        self.csv_dir = os.path.expanduser(csv_dir)
        self.symbols = symbols

    def _load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.csv_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if len(df) < 60:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Alpha158-style features for a single stock."""
        o = df["open"]
        h = df["high"]
        l = df["low"]  # noqa: E741
        c = df["close"]
        v = df["volume"]

        features = pd.DataFrame(index=df.index)

        # ---------- K-line features ----------
        features["KMID"] = (c - o) / o
        features["KLEN"] = (h - l) / o
        features["KMID2"] = np.where(h != l, (c - o) / (h - l), 0)
        features["KUP"] = np.where(h != l, (h - np.maximum(o, c)) / (h - l), 0)
        features["KUP2"] = np.where(h != l, (h - np.maximum(o, c)) / o, 0)
        features["KLOW"] = np.where(h != l, (np.minimum(o, c) - l) / (h - l), 0)
        features["KLOW2"] = np.where(h != l, (np.minimum(o, c) - l) / o, 0)
        features["KSFT"] = (2 * c - h - l) / o
        features["KSFT2"] = np.where(h != l, (2 * c - h - l) / (h - l), 0)

        # ---------- Price relative features ----------
        for w in self.WINDOWS:
            # Open/Close/High/Low relative to close
            features[f"OPEN{w}"] = o / c.shift(w) - 1
            features[f"HIGH{w}"] = h / c.shift(w) - 1
            features[f"LOW{w}"] = l / c.shift(w) - 1
            features[f"CLOSE{w}"] = c / c.shift(w) - 1

        # ---------- Volume features ----------
        for w in self.WINDOWS:
            v_ma = v.rolling(w).mean()
            features[f"VOLUME{w}"] = v / (v_ma + 1e-12) - 1

        # ---------- Moving average features ----------
        for w in self.WINDOWS:
            ma = c.rolling(w).mean()
            features[f"MA{w}"] = c / ma - 1

        # ---------- Standard deviation ----------
        returns = c.pct_change()
        for w in self.WINDOWS:
            features[f"STD{w}"] = returns.rolling(w).std()

        # ---------- Momentum / ROC ----------
        for w in self.WINDOWS:
            features[f"ROC{w}"] = c / c.shift(w) - 1

        # ---------- Max/Min relative ----------
        for w in self.WINDOWS:
            features[f"MAX{w}"] = c / c.rolling(w).max() - 1
            features[f"MIN{w}"] = c / c.rolling(w).min() - 1

        # ---------- Quantile ----------
        for w in self.WINDOWS:
            features[f"QTLU{w}"] = c / c.rolling(w).quantile(0.8) - 1
            features[f"QTLD{w}"] = c / c.rolling(w).quantile(0.2) - 1

        # ---------- Rank ----------
        for w in [5, 10]:
            roll = c.rolling(w)
            features[f"RANK{w}"] = roll.apply(lambda x: pd.Series(x).rank().iloc[-1] / w, raw=False)

        # ---------- RSV (Raw Stochastic Value) ----------
        for w in self.WINDOWS:
            hh = h.rolling(w).max()
            ll = l.rolling(w).min()
            features[f"RSV{w}"] = np.where(hh != ll, (c - ll) / (hh - ll), 0.5)

        # ---------- Correlation ----------
        for w in [5, 10, 20]:
            features[f"CORR{w}"] = c.rolling(w).corr(v.rolling(w).mean())

        # ---------- CORD (corr of daily return and volume change) ----------
        vol_change = v.pct_change()
        for w in [5, 10, 20]:
            features[f"CORD{w}"] = returns.rolling(w).corr(vol_change)

        # ---------- CNTP/CNTN/CNTD ----------
        for w in self.WINDOWS:
            pos = (returns > 0).astype(float)
            neg = (returns < 0).astype(float)
            features[f"CNTP{w}"] = pos.rolling(w).sum() / w
            features[f"CNTN{w}"] = neg.rolling(w).sum() / w
            features[f"CNTD{w}"] = features[f"CNTP{w}"] - features[f"CNTN{w}"]

        # ---------- SUMP/SUMN/SUMD ----------
        pos_ret = returns.clip(lower=0)
        neg_ret = (-returns).clip(lower=0)
        for w in self.WINDOWS:
            sp = pos_ret.rolling(w).sum()
            sn = neg_ret.rolling(w).sum()
            features[f"SUMP{w}"] = sp / (sp + sn + 1e-12)
            features[f"SUMN{w}"] = sn / (sp + sn + 1e-12)
            features[f"SUMD{w}"] = features[f"SUMP{w}"] - features[f"SUMN{w}"]

        # ---------- VMA ----------
        for w in self.WINDOWS:
            features[f"VMA{w}"] = v.rolling(w).mean() / (v.rolling(max(w * 2, 60)).mean() + 1e-12)

        # ---------- VSTD ----------
        for w in self.WINDOWS:
            features[f"VSTD{w}"] = v.rolling(w).std() / (v.rolling(w).mean() + 1e-12)

        # ---------- WVMA ----------
        for w in self.WINDOWS:
            abs_ret = returns.abs()
            features[f"WVMA{w}"] = (abs_ret * v).rolling(w).std() / ((abs_ret * v).rolling(w).mean() + 1e-12)

        # ---------- BETA ----------
        for w in [5, 10, 20]:
            features[f"BETA{w}"] = returns.rolling(w).cov(returns.shift(1)) / (returns.shift(1).rolling(w).var() + 1e-12)

        # ---------- RSQR ----------
        for w in [5, 10, 20]:
            corr = returns.rolling(w).corr(returns.shift(1))
            features[f"RSQR{w}"] = corr ** 2

        # ---------- RESI ----------
        for w in [5, 10, 20]:
            corr = returns.rolling(w).corr(returns.shift(1))
            features[f"RESI{w}"] = 1 - corr ** 2

        # ---------- Skewness and Kurtosis ----------
        for w in [5, 10, 20]:
            features[f"SKEW{w}"] = returns.rolling(w).skew()
            features[f"KURT{w}"] = returns.rolling(w).kurt()

        return features

    def build_dataset(
        self,
        label_days: int = 2,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Build the full dataset with Alpha158 features + label.

        Returns:
            DataFrame with MultiIndex (date, symbol), columns = features + LABEL0
        """
        all_data = []

        print(f"Building Alpha158 dataset for {len(self.symbols)} symbols...")

        for i, symbol in enumerate(self.symbols):
            df = self._load_stock_data(symbol)
            if df is None:
                continue

            features = self._compute_features(df)

            # Label: future N-day return
            features["LABEL0"] = df["close"].shift(-label_days) / df["close"].shift(-1) - 1

            features["date"] = df["date"]
            features["symbol"] = symbol

            all_data.append(features)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(self.symbols)} symbols")

        if not all_data:
            raise ValueError("No data processed. Check CSV directory and symbols.")

        combined = pd.concat(all_data, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])

        # Set multi-index
        combined = combined.set_index(["date", "symbol"])
        combined = combined.sort_index()

        if dropna:
            # Drop rows where label is NaN
            combined = combined.dropna(subset=["LABEL0"])
            # Fill remaining NaN in features with 0
            combined = combined.fillna(0)

        # Replace inf values
        combined = combined.replace([np.inf, -np.inf], 0)

        feature_cols = [c for c in combined.columns if c != "LABEL0"]
        print(f"\nDataset built:")
        print(f"  Shape: {combined.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Date range: {combined.index.get_level_values('date').min()} ~ {combined.index.get_level_values('date').max()}")
        print(f"  Symbols: {combined.index.get_level_values('symbol').nunique()}")

        return combined


def save_dataset(dataset: pd.DataFrame, path: str) -> None:
    """Save dataset as parquet."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_parquet(path)
    print(f"Dataset saved to {path}")


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from parquet."""
    path = os.path.expanduser(path)
    df = pd.read_parquet(path)
    print(f"Dataset loaded from {path}: {df.shape}")
    return df


def split_dataset(
    dataset: pd.DataFrame,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/valid/test based on date ranges.
    """
    dates = dataset.index.get_level_values("date")

    train = dataset[(dates >= train_start) & (dates <= train_end)]
    valid = dataset[(dates >= valid_start) & (dates <= valid_end)]
    test = dataset[(dates >= test_start) & (dates <= test_end)]

    print(f"Dataset split:")
    print(f"  Train: {train.shape} ({train_start} ~ {train_end})")
    print(f"  Valid: {valid.shape} ({valid_start} ~ {valid_end})")
    print(f"  Test:  {test.shape}  ({test_start} ~ {test_end})")

    return train, valid, test
