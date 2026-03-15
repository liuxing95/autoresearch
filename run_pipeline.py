#!/usr/bin/env python3
"""
NASDAQ Quantitative Trading System - Main Pipeline

Usage:
    python run_pipeline.py --step all          # Full pipeline
    python run_pipeline.py --step data         # Data collection only
    python run_pipeline.py --step train        # Train model only
    python run_pipeline.py --step backtest     # Backtest only
    python run_pipeline.py --step analysis     # Analysis & report only

    python run_pipeline.py --model lightgbm    # Use LightGBM (default)
    python run_pipeline.py --model lstm        # Use LSTM
    python run_pipeline.py --model gru         # Use GRU
    python run_pipeline.py --model linear      # Use Ridge regression
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nasdaq_quant.config import (
    PipelineConfig, DataConfig, ModelConfig, StrategyConfig, EnsembleConfig,
    RollingTrainConfig,
    NASDAQ100_SYMBOLS, RESULTS_DIR, MODEL_DIR, REPORT_DIR,
)
from nasdaq_quant.data.collector import download_stock_data, download_benchmark
from nasdaq_quant.data.processor import (
    Alpha158Builder, csv_to_qlib_format,
    save_dataset, load_dataset, split_dataset,
)
from nasdaq_quant.models.trainer import create_model, prepare_xy
from nasdaq_quant.strategy.backtest import (
    TopKDropoutStrategy, BacktestEngine,
    compute_daily_returns, compute_benchmark_returns,
)
from nasdaq_quant.analysis.report import (
    compute_ic_series, group_return_analysis,
    generate_report, plot_results, plot_equity_curve,
)
from typing import Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_DIR = os.path.expanduser("~/.qlib/csv_data/nasdaq")
DATASET_PATH = os.path.join(RESULTS_DIR, "dataset.parquet")
PRED_PATH = os.path.join(RESULTS_DIR, "predictions.parquet")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_result.pkl")
RESULTS_TSV = os.path.join(RESULTS_DIR, "results.tsv")


# ---------------------------------------------------------------------------
# Step 1: Data Collection
# ---------------------------------------------------------------------------

def step_data(config: PipelineConfig) -> None:
    """Download stock data and prepare dataset."""
    print("\n" + "=" * 60)
    print("  Step 1: Data Collection & Preparation")
    print("=" * 60)

    t0 = time.time()

    # 1a. Download NASDAQ 100 stock data
    print("\n[1/3] Downloading NASDAQ 100 stock data...")
    download_stock_data(
        symbols=config.data.symbols,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        output_dir=CSV_DIR,
    )

    # 1b. Download benchmark
    print("\n[2/3] Downloading benchmark (NASDAQ 100 Index)...")
    download_benchmark(
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        output_dir=CSV_DIR,
    )

    # 1c. Build Alpha158 dataset
    print("\n[3/3] Building Alpha158 feature dataset...")
    builder = Alpha158Builder(csv_dir=CSV_DIR, symbols=config.data.symbols)
    dataset = builder.build_dataset(label_days=2)

    save_dataset(dataset, DATASET_PATH)

    # Also convert to Qlib format for reference
    csv_to_qlib_format(
        csv_dir=CSV_DIR,
        qlib_dir=config.data.data_dir,
        symbols=config.data.symbols,
    )

    dt = time.time() - t0
    print(f"\nData preparation complete in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Step 2: Model Training
# ---------------------------------------------------------------------------

def step_train(config: PipelineConfig) -> None:
    """Train prediction model."""
    print("\n" + "=" * 60)
    print("  Step 2: Model Training")
    print("=" * 60)

    t0 = time.time()

    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # Split
    train_data, valid_data, test_data = split_dataset(
        dataset,
        config.model.train_start, config.model.train_end,
        config.model.valid_start, config.model.valid_end,
        config.model.test_start, config.model.test_end,
    )

    # Feature normalization (cross-sectional z-score per date)
    feature_cols = [c for c in dataset.columns if c != config.model.label_col]

    def cross_sectional_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Z-score normalize features cross-sectionally (per date)."""
        df = df.copy()
        grouped = df.groupby(level="date")
        for col in cols:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            df[col] = (df[col] - mean) / (std + 1e-12)
        return df

    print("\nNormalizing features (cross-sectional z-score)...")
    train_norm = cross_sectional_normalize(train_data, feature_cols)
    valid_norm = cross_sectional_normalize(valid_data, feature_cols)
    test_norm = cross_sectional_normalize(test_data, feature_cols)

    # Replace inf/nan after normalization
    for df in [train_norm, valid_norm, test_norm]:
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    # Create model
    model_type = config.model.model_type
    print(f"\nModel type: {model_type}")

    if model_type == "lightgbm":
        model = create_model(
            "lightgbm",
            n_estimators=config.model.lgb.n_estimators,
            learning_rate=config.model.lgb.learning_rate,
            max_depth=config.model.lgb.max_depth,
            num_leaves=config.model.lgb.num_leaves,
            subsample=config.model.lgb.subsample,
            colsample_bytree=config.model.lgb.colsample_bytree,
            lambda_l1=config.model.lgb.lambda_l1,
            lambda_l2=config.model.lgb.lambda_l2,
            early_stopping_rounds=config.model.lgb.early_stopping_rounds,
            num_threads=config.model.lgb.num_threads,
        )
    elif model_type == "linear":
        model = create_model("linear")
    elif model_type in ("lstm", "gru"):
        model = create_model(
            model_type,
            d_feat=len(feature_cols),
            hidden_size=config.model.lstm.hidden_size,
            num_layers=config.model.lstm.num_layers,
            dropout=config.model.lstm.dropout,
            n_epochs=config.model.lstm.n_epochs,
            lr=config.model.lstm.lr,
            early_stop=config.model.lstm.early_stop,
            batch_size=config.model.lstm.batch_size,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Train
    metrics = model.train(train_norm, valid_norm, label_col=config.model.label_col)

    # Save model
    model_ext = ".pt" if model_type in ("lstm", "gru") else ".pkl"
    model_path = os.path.join(MODEL_DIR, f"{model_type}_model{model_ext}")
    model.save(model_path)

    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_preds = model.predict(test_norm, label_col=config.model.label_col)

    # Also predict on validation set for IC analysis
    valid_preds = model.predict(valid_norm, label_col=config.model.label_col)

    # Combine predictions
    all_preds = pd.concat([valid_preds, test_preds])
    all_preds.to_frame("pred_score").to_parquet(PRED_PATH)

    # Feature importance (for tree models)
    if model_type == "lightgbm":
        fi = model.feature_importance(feature_names=feature_cols)
        fi_path = os.path.join(REPORT_DIR, "feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        print(f"\nTop 20 features:")
        print(fi.head(20).to_string())

    # Log to results.tsv
    _log_result(config, model_type, metrics)

    dt = time.time() - t0
    print(f"\nModel training complete in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Step 2b: Rolling Walk-Forward Training
# ---------------------------------------------------------------------------

def _select_features_by_ic(
    train_data: pd.DataFrame,
    label_col: str = "LABEL0",
    n_features: int = 80,
) -> list:
    """Select top features by IC stability (mean IC / std IC)."""
    feature_cols = [c for c in train_data.columns if c != label_col]
    ic_scores = {}

    # Compute daily IC for each feature
    dates = train_data.index.get_level_values("date").unique()
    for feat in feature_cols:
        daily_ics = []
        for dt in dates:
            try:
                day_data = train_data.xs(dt, level="date")
                if len(day_data) < 10:
                    continue
                ic = np.corrcoef(day_data[feat].values, day_data[label_col].values)[0, 1]
                if np.isfinite(ic):
                    daily_ics.append(ic)
            except Exception:
                continue

        if len(daily_ics) > 20:
            mean_ic = np.mean(daily_ics)
            std_ic = np.std(daily_ics) + 1e-12
            icir = abs(mean_ic) / std_ic  # Use ICIR as stability measure
            ic_scores[feat] = icir

    # Sort by ICIR and take top N
    sorted_feats = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in sorted_feats[:n_features]]

    if len(selected) < 10:
        # Fallback: use all features if too few selected
        return feature_cols

    return selected


def step_rolling_train(config: PipelineConfig) -> None:
    """
    Rolling walk-forward training: retrain periodically with recent data.

    Instead of training once on 2015-2021 and testing forever, this approach:
    1. Divides the test period into rolling windows
    2. For each window, trains on the most recent N years of data
    3. Validates on a recent buffer period
    4. Generates predictions for that window only
    5. Concatenates all predictions for the full test period
    """
    print("\n" + "=" * 60)
    print("  Step 2b: Rolling Walk-Forward Training")
    print("=" * 60)

    t0 = time.time()
    rolling = config.rolling

    # Load full dataset
    dataset = load_dataset(DATASET_PATH)
    feature_cols = [c for c in dataset.columns if c != config.model.label_col]
    label_col = config.model.label_col

    # Parse test period
    test_start = pd.Timestamp(config.model.test_start)
    test_end = pd.Timestamp(config.model.test_end)

    # Generate rolling windows
    windows = []
    current_start = test_start
    while current_start < test_end:
        window_end = min(
            current_start + pd.DateOffset(months=rolling.retrain_months),
            test_end
        )

        # Train period: train_window_years before the test window
        train_start = current_start - pd.DateOffset(years=rolling.train_window_years)
        # Validation: valid_months before test window with purge gap
        valid_end = current_start - pd.DateOffset(days=rolling.purge_days)
        valid_start = valid_end - pd.DateOffset(months=rolling.valid_months)

        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": (valid_start - pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            "valid_start": valid_start.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
            "test_start": current_start.strftime("%Y-%m-%d"),
            "test_end": window_end.strftime("%Y-%m-%d"),
        })
        current_start = window_end

    print(f"\n  Rolling windows: {len(windows)}")
    print(f"  Train window:    {rolling.train_window_years} years")
    print(f"  Valid window:    {rolling.valid_months} months")
    print(f"  Retrain every:   {rolling.retrain_months} months")
    print(f"  Feature select:  {rolling.feature_selection} (top {rolling.n_features})")

    def cross_sectional_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby(level="date")
        for col in cols:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            df[col] = (df[col] - mean) / (std + 1e-8)
        # Clip extreme z-scores to prevent overflow in model computation
        df[cols] = df[cols].clip(-5, 5)
        df[cols] = df[cols].fillna(0)
        return df

    all_test_preds = []
    all_valid_preds = []
    window_metrics = []

    for i, w in enumerate(windows):
        print(f"\n{'─'*50}")
        print(f"  Window {i+1}/{len(windows)}")
        print(f"  Train: {w['train_start']} ~ {w['train_end']}")
        print(f"  Valid: {w['valid_start']} ~ {w['valid_end']}")
        print(f"  Test:  {w['test_start']} ~ {w['test_end']}")
        print(f"{'─'*50}")

        # Split data for this window
        dates = dataset.index.get_level_values("date")
        train_data = dataset[(dates >= w["train_start"]) & (dates <= w["train_end"])]
        valid_data = dataset[(dates >= w["valid_start"]) & (dates <= w["valid_end"])]
        test_data = dataset[(dates >= w["test_start"]) & (dates <= w["test_end"])]

        if len(train_data) < 100 or len(valid_data) < 50 or len(test_data) == 0:
            print(f"  Skipping window: insufficient data "
                  f"(train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)})")
            continue

        # Feature selection
        used_features = feature_cols
        if rolling.feature_selection:
            print(f"  Selecting top {rolling.n_features} features by IC stability...")
            used_features = _select_features_by_ic(
                train_data, label_col, rolling.n_features
            )
            print(f"  Selected {len(used_features)} features")
            # Keep only selected features + label
            train_data = train_data[used_features + [label_col]]
            valid_data = valid_data[used_features + [label_col]]
            test_data = test_data[used_features + [label_col]]

        # Normalize
        train_norm = cross_sectional_normalize(train_data, used_features)
        valid_norm = cross_sectional_normalize(valid_data, used_features)
        test_norm = cross_sectional_normalize(test_data, used_features)

        for df in [train_norm, valid_norm, test_norm]:
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

        # Create and train model
        model_type = config.model.model_type
        if model_type == "lightgbm":
            model = create_model(
                "lightgbm",
                n_estimators=config.model.lgb.n_estimators,
                learning_rate=config.model.lgb.learning_rate,
                max_depth=config.model.lgb.max_depth,
                num_leaves=config.model.lgb.num_leaves,
                subsample=config.model.lgb.subsample,
                colsample_bytree=config.model.lgb.colsample_bytree,
                lambda_l1=config.model.lgb.lambda_l1,
                lambda_l2=config.model.lgb.lambda_l2,
                early_stopping_rounds=config.model.lgb.early_stopping_rounds,
                num_threads=config.model.lgb.num_threads,
            )
        elif model_type == "linear":
            model = create_model("linear")
        else:
            model = create_model(model_type, d_feat=len(used_features))

        metrics = model.train(train_norm, valid_norm, label_col=label_col)
        window_metrics.append(metrics)

        # Generate predictions
        test_preds = model.predict(test_norm, label_col=label_col)
        valid_preds = model.predict(valid_norm, label_col=label_col)
        all_test_preds.append(test_preds)
        all_valid_preds.append(valid_preds)

        print(f"  Window IC: train={metrics.get('train_ic', 0):.4f}, "
              f"valid={metrics.get('valid_ic', 0):.4f}")

    # Concatenate all predictions
    if all_test_preds:
        combined_test = pd.concat(all_test_preds)
        # Remove duplicates (overlapping dates between windows)
        combined_test = combined_test[~combined_test.index.duplicated(keep='last')]

        combined_valid = pd.concat(all_valid_preds) if all_valid_preds else pd.Series(dtype=float)
        if not combined_valid.empty:
            combined_valid = combined_valid[~combined_valid.index.duplicated(keep='last')]

        all_preds = pd.concat([combined_valid, combined_test])
        all_preds = all_preds[~all_preds.index.duplicated(keep='last')]
        all_preds.to_frame("pred_score").to_parquet(PRED_PATH)

        # Summary
        avg_train_ic = np.mean([m.get("train_ic", 0) for m in window_metrics])
        avg_valid_ic = np.mean([m.get("valid_ic", 0) for m in window_metrics])
        print(f"\n{'='*50}")
        print(f"  Rolling training summary:")
        print(f"  Windows trained: {len(window_metrics)}")
        print(f"  Avg train IC:    {avg_train_ic:.4f}")
        print(f"  Avg valid IC:    {avg_valid_ic:.4f}")
        print(f"  Total preds:     {len(combined_test)}")

        # Log
        _log_result(config, f"rolling_{config.model.model_type}",
                    {"valid_ic": avg_valid_ic, "valid_mse": 0})
    else:
        print("\nERROR: No predictions generated. Check data availability.")

    dt = time.time() - t0
    print(f"\nRolling training complete in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Step 3: Backtest
# ---------------------------------------------------------------------------

def step_backtest(config: PipelineConfig) -> None:
    """Run backtesting."""
    print("\n" + "=" * 60)
    print("  Step 3: Backtesting")
    print("=" * 60)

    t0 = time.time()

    # Load predictions
    pred_df = pd.read_parquet(PRED_PATH)
    pred_scores = pred_df["pred_score"]

    # Filter to test period only
    dates = pred_scores.index.get_level_values("date")
    test_preds = pred_scores[(dates >= config.model.test_start) & (dates <= config.model.test_end)]

    if test_preds.empty:
        print("No predictions in test period. Make sure training was done.")
        return

    print(f"Test predictions: {len(test_preds)} (date x symbol pairs)")

    # Generate strategy signals
    print("\nGenerating trading signals...")
    # Auto-cap topk to number of available symbols
    n_symbols = test_preds.index.get_level_values("symbol").nunique()
    effective_topk = min(config.strategy.topk, max(n_symbols - 1, 1))
    effective_n_drop = min(config.strategy.n_drop, max(effective_topk // 5, 1))
    if effective_topk != config.strategy.topk:
        print(f"  Auto-adjusted topk: {config.strategy.topk} -> {effective_topk} (only {n_symbols} symbols)")
    strategy = TopKDropoutStrategy(
        topk=effective_topk,
        n_drop=effective_n_drop,
        max_weight=config.strategy.max_weight,
        rebalance_freq=config.strategy.rebalance_freq,
    )
    signals = strategy.generate_signals(test_preds)
    print(f"Trading signals: {len(signals)} entries over {signals['date'].nunique()} days")

    # Compute daily returns
    print("\nComputing daily returns...")
    returns_data = compute_daily_returns(
        csv_dir=CSV_DIR,
        symbols=config.data.symbols,
        start_date=config.model.test_start,
        end_date=config.model.test_end,
    )

    # Run backtest
    print("\nRunning backtest simulation...")
    print(f"  Stop Loss: {config.strategy.stop_loss*100:.0f}%")
    print(f"  Drawdown Limit: {config.strategy.max_drawdown_limit*100:.0f}%")
    print(f"  Slippage: {config.strategy.slippage*100:.1f}%")
    engine = BacktestEngine(
        init_cash=config.strategy.init_cash,
        buy_cost=config.strategy.buy_cost,
        sell_cost=config.strategy.sell_cost,
        slippage=config.strategy.slippage,
        stop_loss=config.strategy.stop_loss,
        max_drawdown_limit=config.strategy.max_drawdown_limit,
    )
    result = engine.run(signals, returns_data)

    # Print summary
    metrics = result["metrics"]
    print("\n--- Backtest Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "return" in k or "drawdown" in k or "rate" in k or "turnover" in k:
                print(f"  {k:25s}: {v*100:.2f}%")
            elif "equity" in k:
                print(f"  {k:25s}: ${v:,.2f}")
            else:
                print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    # Save backtest result
    with open(BACKTEST_PATH, "wb") as f:
        pickle.dump(result, f)

    dt = time.time() - t0
    print(f"\nBacktest complete in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Step: Model Ensemble
# ---------------------------------------------------------------------------

def step_ensemble(config: PipelineConfig) -> None:
    """Train multiple models and generate ensemble predictions."""
    print("\n" + "=" * 60)
    print("  Step: Model Ensemble")
    print("=" * 60)

    t0 = time.time()

    from nasdaq_quant.models.ensemble import ModelEnsemble

    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # Split
    train_data, valid_data, test_data = split_dataset(
        dataset,
        config.model.train_start, config.model.train_end,
        config.model.valid_start, config.model.valid_end,
        config.model.test_start, config.model.test_end,
    )

    # Feature normalization
    feature_cols = [c for c in dataset.columns if c != config.model.label_col]

    def cross_sectional_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby(level="date")
        for col in cols:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            df[col] = (df[col] - mean) / (std + 1e-12)
        return df

    train_norm = cross_sectional_normalize(train_data, feature_cols)
    valid_norm = cross_sectional_normalize(valid_data, feature_cols)
    test_norm = cross_sectional_normalize(test_data, feature_cols)

    for df in [train_norm, valid_norm, test_norm]:
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

    # Build model configs for ensemble
    model_configs = []
    for model_type in config.ensemble.models:
        if model_type == "lightgbm":
            model_configs.append({
                "type": "lightgbm",
                "n_estimators": config.model.lgb.n_estimators,
                "learning_rate": config.model.lgb.learning_rate,
                "max_depth": config.model.lgb.max_depth,
                "num_leaves": config.model.lgb.num_leaves,
                "subsample": config.model.lgb.subsample,
                "colsample_bytree": config.model.lgb.colsample_bytree,
                "lambda_l1": config.model.lgb.lambda_l1,
                "lambda_l2": config.model.lgb.lambda_l2,
                "early_stopping_rounds": config.model.lgb.early_stopping_rounds,
                "num_threads": config.model.lgb.num_threads,
            })
        elif model_type == "linear":
            model_configs.append({"type": "linear"})
        elif model_type in ("lstm", "gru"):
            model_configs.append({
                "type": model_type,
                "d_feat": len(feature_cols),
                "hidden_size": config.model.lstm.hidden_size,
                "num_layers": config.model.lstm.num_layers,
                "dropout": config.model.lstm.dropout,
                "n_epochs": config.model.lstm.n_epochs,
                "lr": config.model.lstm.lr,
                "early_stop": config.model.lstm.early_stop,
                "batch_size": config.model.lstm.batch_size,
            })

    # Create and train ensemble
    ensemble = ModelEnsemble(
        model_configs=model_configs,
        method=config.ensemble.method,
    )

    metrics = ensemble.train(
        train_norm, valid_norm,
        label_col=config.model.label_col,
        feature_cols=feature_cols,
    )

    # Save ensemble
    ensemble.save(MODEL_DIR)

    # Generate predictions
    print("\nGenerating ensemble predictions...")
    test_preds = ensemble.predict(test_norm, label_col=config.model.label_col)
    valid_preds = ensemble.predict(valid_norm, label_col=config.model.label_col)

    all_preds = pd.concat([valid_preds, test_preds])
    all_preds.to_frame("pred_score").to_parquet(PRED_PATH)

    dt = time.time() - t0
    print(f"\nEnsemble training complete in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Step 4: Analysis & Reporting
# ---------------------------------------------------------------------------

def step_analysis(config: PipelineConfig) -> None:
    """Generate analysis reports and visualizations."""
    print("\n" + "=" * 60)
    print("  Step 4: Analysis & Reporting")
    print("=" * 60)

    # Load backtest result
    if not os.path.exists(BACKTEST_PATH):
        print("No backtest result found. Run backtest first.")
        return

    with open(BACKTEST_PATH, "rb") as f:
        backtest_result = pickle.load(f)

    # Load predictions and actual returns
    pred_df = pd.read_parquet(PRED_PATH)
    pred_scores = pred_df["pred_score"]

    # Load dataset for actual labels
    dataset = load_dataset(DATASET_PATH)
    actual_returns = dataset["LABEL0"]

    # Filter to test period
    dates_pred = pred_scores.index.get_level_values("date")
    test_preds = pred_scores[(dates_pred >= config.model.test_start)]
    dates_actual = actual_returns.index.get_level_values("date")
    test_actuals = actual_returns[(dates_actual >= config.model.test_start)]

    # IC analysis
    print("\nComputing IC analysis...")
    ic_df = compute_ic_series(test_preds, test_actuals)
    print(f"IC computed for {len(ic_df)} trading days")

    # Group return analysis
    print("\nComputing group return analysis...")
    group_df = group_return_analysis(test_preds, test_actuals, n_groups=5)

    # Benchmark returns
    bench_returns = compute_benchmark_returns(
        csv_dir=CSV_DIR,
        start_date=config.model.test_start,
        end_date=config.model.test_end,
    )

    # Generate report
    report = generate_report(
        backtest_result,
        ic_df=ic_df,
        group_df=group_df,
        output_dir=REPORT_DIR,
    )
    print("\n" + report)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_results(
        backtest_result,
        ic_df=ic_df,
        benchmark_returns=bench_returns,
        output_dir=REPORT_DIR,
    )
    plot_equity_curve(
        backtest_result,
        benchmark_returns=bench_returns,
        output_dir=REPORT_DIR,
    )

    # Save IC data
    if not ic_df.empty:
        ic_path = os.path.join(REPORT_DIR, "ic_analysis.csv")
        ic_df.to_csv(ic_path, index=False)
        print(f"IC data saved to {ic_path}")

    # Save group analysis
    if not group_df.empty:
        group_path = os.path.join(REPORT_DIR, "group_analysis.csv")
        group_df.to_csv(group_path, index=False)

    print("\nAnalysis complete!")


# ---------------------------------------------------------------------------
# Utility: Log results
# ---------------------------------------------------------------------------

def _log_result(config: PipelineConfig, model_type: str, metrics: Dict) -> None:
    """Log experiment result to results.tsv."""
    import datetime

    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("timestamp\tmodel\tvalid_ic\tvalid_mse\tstatus\tdescription\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    valid_ic = metrics.get("valid_ic", 0)
    valid_mse = metrics.get("valid_mse", 0)

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{ts}\t{model_type}\t{valid_ic:.6f}\t{valid_mse:.6f}\tkeep\tbaseline {model_type}\n")

    print(f"Result logged to {RESULTS_TSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NASDAQ Quantitative Trading Pipeline")
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "data", "train", "backtest", "analysis", "ensemble", "rolling_train"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--model", type=str, default="lightgbm",
        choices=["lightgbm", "linear", "lstm", "gru"],
        help="Model type to use",
    )
    parser.add_argument(
        "--topk", type=int, default=30,
        help="Number of stocks to hold (TopK strategy)",
    )
    parser.add_argument(
        "--n-drop", type=int, default=5,
        help="Max stocks to drop per rebalance",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated list of symbols (default: NASDAQ 100)",
    )
    parser.add_argument(
        "--test-start", type=str, default="2021-01-01",
        help="Test period start date",
    )
    parser.add_argument(
        "--test-end", type=str, default="2022-12-31",
        help="Test period end date",
    )
    # LightGBM hyperparams
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="LightGBM learning rate")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="LightGBM max depth")
    parser.add_argument("--num-leaves", type=int, default=None,
                        help="LightGBM num leaves")
    parser.add_argument("--lambda-l1", type=float, default=None,
                        help="LightGBM L1 regularization")
    parser.add_argument("--lambda-l2", type=float, default=None,
                        help="LightGBM L2 regularization")
    # Risk control
    parser.add_argument("--stop-loss", type=float, default=-0.08,
                        help="Per-stock stop loss threshold (e.g. -0.08)")
    parser.add_argument("--max-drawdown-limit", type=float, default=-0.20,
                        help="Portfolio drawdown circuit breaker")
    parser.add_argument("--rebalance-freq", type=str, default="day",
                        choices=["day", "week"], help="Rebalancing frequency")
    # Ensemble
    parser.add_argument("--ensemble", action="store_true", default=False,
                        help="Enable model ensemble (lightgbm + linear)")
    parser.add_argument("--ensemble-method", type=str, default="ic_weighted",
                        choices=["equal", "ic_weighted", "rank_average"],
                        help="Ensemble weighting method")
    # Rolling walk-forward training
    parser.add_argument("--rolling", action="store_true", default=False,
                        help="Enable rolling walk-forward training")
    parser.add_argument("--train-window", type=int, default=3,
                        help="Rolling training window in years (default: 3)")
    parser.add_argument("--retrain-freq", type=int, default=6,
                        help="Retrain frequency in months (default: 6)")
    parser.add_argument("--feature-select", action="store_true", default=False,
                        help="Enable IC-based feature selection")
    parser.add_argument("--n-features", type=int, default=80,
                        help="Number of features to select (default: 80)")
    parser.add_argument("--label-days", type=int, default=2,
                        help="Forward return horizon in days (default: 2)")
    parser.add_argument("--subsample", type=float, default=None,
                        help="LightGBM subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=None,
                        help="LightGBM column sample ratio")

    args = parser.parse_args()

    # Build configuration
    config = PipelineConfig()
    config.model.model_type = args.model
    config.strategy.topk = args.topk
    config.strategy.n_drop = args.n_drop
    config.strategy.stop_loss = args.stop_loss
    config.strategy.max_drawdown_limit = args.max_drawdown_limit
    config.strategy.rebalance_freq = args.rebalance_freq
    config.model.test_start = args.test_start
    config.model.test_end = args.test_end

    # LightGBM hyperparam overrides
    if args.learning_rate is not None:
        config.model.lgb.learning_rate = args.learning_rate
    if args.max_depth is not None:
        config.model.lgb.max_depth = args.max_depth
    if args.num_leaves is not None:
        config.model.lgb.num_leaves = args.num_leaves
    if args.lambda_l1 is not None:
        config.model.lgb.lambda_l1 = args.lambda_l1
    if args.lambda_l2 is not None:
        config.model.lgb.lambda_l2 = args.lambda_l2

    # Ensemble config
    config.ensemble.enable = args.ensemble
    config.ensemble.method = args.ensemble_method

    # Rolling training config
    config.rolling.enable = args.rolling
    config.rolling.train_window_years = args.train_window
    config.rolling.retrain_months = args.retrain_freq
    config.rolling.feature_selection = args.feature_select
    config.rolling.n_features = args.n_features
    config.rolling.label_days = args.label_days

    # LightGBM subsample and colsample overrides
    if args.subsample is not None:
        config.model.lgb.subsample = args.subsample
    if args.colsample_bytree is not None:
        config.model.lgb.colsample_bytree = args.colsample_bytree

    if args.symbols:
        config.data.symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 60)
    print("  NASDAQ Quantitative Trading System")
    print("=" * 60)
    print(f"  Model:      {config.model.model_type}{' (ensemble)' if config.ensemble.enable else ''}{' (rolling)' if config.rolling.enable else ''}")
    print(f"  TopK:       {config.strategy.topk}")
    print(f"  Rebalance:  {config.strategy.rebalance_freq}")
    print(f"  Stop Loss:  {config.strategy.stop_loss*100:.0f}%")
    print(f"  DD Limit:   {config.strategy.max_drawdown_limit*100:.0f}%")
    if config.rolling.enable:
        print(f"  Train Win:  {config.rolling.train_window_years}yr, retrain every {config.rolling.retrain_months}mo")
        print(f"  Feat Sel:   {config.rolling.feature_selection} (top {config.rolling.n_features})")
    print(f"  Symbols:    {len(config.data.symbols)} stocks")
    print(f"  Train:      {config.model.train_start} ~ {config.model.train_end}")
    print(f"  Valid:      {config.model.valid_start} ~ {config.model.valid_end}")
    print(f"  Test:       {config.model.test_start} ~ {config.model.test_end}")
    print(f"  Results:    {RESULTS_DIR}")
    print("=" * 60)

    t_total = time.time()

    if args.step in ("all", "data"):
        step_data(config)

    if args.step == "rolling_train" or (args.step == "all" and config.rolling.enable):
        step_rolling_train(config)
    elif args.step in ("all", "train"):
        step_train(config)

    if args.step == "ensemble" or (args.step == "all" and config.ensemble.enable):
        step_ensemble(config)

    if args.step in ("all", "backtest"):
        step_backtest(config)

    if args.step in ("all", "analysis"):
        step_analysis(config)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
