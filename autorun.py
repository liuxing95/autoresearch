#!/usr/bin/env python3
"""
AutoQuant: Autonomous Experiment Loop

Emulates the autoresearch paradigm for quantitative trading:
  1. Read current best results
  2. Mutate strategy parameters (model, hyperparams, topk, etc.)
  3. Run backtest
  4. Compare sharpe_ratio — keep or discard
  5. Log to results.tsv
  6. Loop

Usage:
    python autorun.py                     # Run indefinitely
    python autorun.py --max-iter 20       # Run 20 experiments
    python autorun.py --symbols AAPL,MSFT,GOOGL   # Custom symbols
"""

import os
import sys
import time
import random
import argparse
import subprocess
import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_TSV = os.path.join("quant_results", "autorun_results.tsv")


# ---------------------------------------------------------------------------
# Experiment parameter space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "model": ["lightgbm", "linear"],
    "topk": [5, 10, 15, 20, 30],
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.005],
    "max_depth": [4, 5, 6, 7, 8],
    "num_leaves": [31, 48, 64, 96, 128],
    "lambda_l1": [0.0, 0.1, 0.5, 1.0, 5.0],
    "lambda_l2": [0.0, 0.1, 0.5, 1.0, 5.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "n_drop": [1, 2, 3, 5, 8],
    "stop_loss": [-0.05, -0.08, -0.10, -0.15, -1.0],
    "rebalance_freq": ["day", "week"],
}


def load_results() -> list:
    """Load previous experiment results."""
    results = []
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    results.append({
                        "timestamp": parts[0],
                        "sharpe": float(parts[1]),
                        "max_dd": float(parts[2]),
                        "annual_ret": float(parts[3]),
                        "status": parts[4],
                        "model": parts[5],
                        "description": parts[6],
                    })
    return results


def get_best_sharpe(results: list) -> float:
    """Get the best sharpe ratio from kept experiments."""
    kept = [r for r in results if r["status"] == "keep"]
    if not kept:
        return -999.0
    return max(r["sharpe"] for r in kept)


def mutate_params(results: list) -> dict:
    """Generate a new set of experiment parameters by mutation."""
    params = {
        "model": random.choice(SEARCH_SPACE["model"]),
        "topk": random.choice(SEARCH_SPACE["topk"]),
        "n_drop": random.choice(SEARCH_SPACE["n_drop"]),
        "stop_loss": random.choice(SEARCH_SPACE["stop_loss"]),
        "rebalance_freq": random.choice(SEARCH_SPACE["rebalance_freq"]),
    }

    # LightGBM-specific params
    if params["model"] == "lightgbm":
        params["learning_rate"] = random.choice(SEARCH_SPACE["learning_rate"])
        params["max_depth"] = random.choice(SEARCH_SPACE["max_depth"])
        params["num_leaves"] = random.choice(SEARCH_SPACE["num_leaves"])
        params["lambda_l1"] = random.choice(SEARCH_SPACE["lambda_l1"])
        params["lambda_l2"] = random.choice(SEARCH_SPACE["lambda_l2"])
        params["subsample"] = random.choice(SEARCH_SPACE["subsample"])
        params["colsample_bytree"] = random.choice(SEARCH_SPACE["colsample_bytree"])

    return params


def run_experiment(params: dict, symbols: str = None) -> dict:
    """
    Run a single experiment by invoking the pipeline.

    Returns dict with sharpe, max_dd, annual_ret, status.
    """
    cmd = [
        sys.executable, "run_pipeline.py",
        "--step", "all",
        "--model", params["model"],
        "--topk", str(params["topk"]),
        "--n-drop", str(params["n_drop"]),
        "--stop-loss", str(params["stop_loss"]),
        "--rebalance-freq", params["rebalance_freq"],
    ]

    if params["model"] == "lightgbm":
        cmd.extend(["--learning-rate", str(params["learning_rate"])])
        cmd.extend(["--max-depth", str(params["max_depth"])])
        cmd.extend(["--num-leaves", str(params["num_leaves"])])
        cmd.extend(["--lambda-l1", str(params["lambda_l1"])])
        cmd.extend(["--lambda-l2", str(params["lambda_l2"])])

    if symbols:
        cmd.extend(["--symbols", symbols])

    print(f"\n{'='*60}")
    print(f"  Running experiment: {params['model']} topk={params['topk']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        output = result.stdout + result.stderr

        # Parse metrics from output
        sharpe = _extract_metric(output, "sharpe_ratio")
        max_dd = _extract_metric(output, "max_drawdown")
        annual_ret = _extract_metric(output, "annualized_return")

        if sharpe is None:
            return {"sharpe": 0, "max_dd": 0, "annual_ret": 0, "status": "crash"}

        # Hard constraint: max drawdown
        if max_dd is not None and max_dd < -0.30:
            return {"sharpe": sharpe, "max_dd": max_dd, "annual_ret": annual_ret or 0, "status": "crash"}

        return {
            "sharpe": sharpe,
            "max_dd": max_dd or 0,
            "annual_ret": annual_ret or 0,
            "status": "pending",
        }

    except subprocess.TimeoutExpired:
        return {"sharpe": 0, "max_dd": 0, "annual_ret": 0, "status": "crash"}
    except Exception as e:
        print(f"  Experiment failed: {e}")
        return {"sharpe": 0, "max_dd": 0, "annual_ret": 0, "status": "crash"}


def _extract_metric(output: str, metric_name: str) -> float:
    """Extract a metric value from pipeline output."""
    import re

    # Primary pattern: "  metric_name             : value%" or "  metric_name             : value"
    # The pipeline outputs lines like:
    #   sharpe_ratio             : 2.0206
    #   max_drawdown             : -8.81%
    #   annualized_return        : 27.40%
    pattern = rf"{metric_name}\s*:\s*([-+]?\d+\.?\d*)(%?)"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        has_pct = match.group(2) == "%"
        if has_pct:
            val = val / 100.0  # Convert percentage to decimal
        return val

    return None


def log_result(params: dict, result: dict) -> None:
    """Log experiment result to TSV file."""
    os.makedirs(os.path.dirname(RESULTS_TSV), exist_ok=True)

    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("timestamp\tsharpe\tmax_dd\tannual_ret\tstatus\tmodel\tdescription\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    desc = _make_description(params)

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{ts}\t{result['sharpe']:.4f}\t{result['max_dd']:.4f}\t"
                f"{result['annual_ret']:.4f}\t{result['status']}\t"
                f"{params['model']}\t{desc}\n")


def _make_description(params: dict) -> str:
    """Create a short description of the experiment."""
    parts = [f"{params['model']}"]
    parts.append(f"topk={params['topk']}")
    parts.append(f"n_drop={params['n_drop']}")
    parts.append(f"freq={params['rebalance_freq']}")
    if params["model"] == "lightgbm":
        parts.append(f"lr={params.get('learning_rate', '?')}")
        parts.append(f"depth={params.get('max_depth', '?')}")
    if params.get("stop_loss", -1.0) > -1.0:
        parts.append(f"sl={params['stop_loss']}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AutoQuant: Autonomous Experiment Loop")
    parser.add_argument("--max-iter", type=int, default=0,
                        help="Max iterations (0 = infinite)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated stock symbols")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("  AutoQuant: Autonomous Experiment Loop")
    print("=" * 60)
    print(f"  Max iterations: {'infinite' if args.max_iter == 0 else args.max_iter}")
    print(f"  Results file:   {RESULTS_TSV}")
    print(f"  Symbols:        {args.symbols or 'NASDAQ 100 (default)'}")
    print("=" * 60)

    iteration = 0
    total_keep = 0
    total_discard = 0
    total_crash = 0

    while True:
        iteration += 1
        if args.max_iter > 0 and iteration > args.max_iter:
            break

        print(f"\n{'#'*60}")
        print(f"  Iteration {iteration}")
        print(f"{'#'*60}")

        # Load historical results
        results = load_results()
        best_sharpe = get_best_sharpe(results)
        print(f"  Historical experiments: {len(results)}")
        print(f"  Best sharpe so far: {best_sharpe:.4f}")

        # Mutate parameters
        params = mutate_params(results)
        print(f"  New params: {params}")

        # Run experiment
        t0 = time.time()
        exp_result = run_experiment(params, symbols=args.symbols)
        dt = time.time() - t0

        # Decide: keep or discard
        if exp_result["status"] == "crash":
            exp_result["status"] = "crash"
            total_crash += 1
            print(f"\n  CRASH — experiment failed or exceeded limits")
        elif exp_result["sharpe"] > best_sharpe:
            exp_result["status"] = "keep"
            total_keep += 1
            print(f"\n  KEEP — sharpe {exp_result['sharpe']:.4f} > {best_sharpe:.4f}")
        else:
            exp_result["status"] = "discard"
            total_discard += 1
            print(f"\n  DISCARD — sharpe {exp_result['sharpe']:.4f} <= {best_sharpe:.4f}")

        # Log
        log_result(params, exp_result)

        print(f"  Time: {dt:.1f}s")
        print(f"  Sharpe: {exp_result['sharpe']:.4f}, MaxDD: {exp_result['max_dd']:.4f}, "
              f"AnnRet: {exp_result['annual_ret']:.4f}")
        print(f"  Totals — Keep: {total_keep}, Discard: {total_discard}, Crash: {total_crash}")

    print(f"\n{'='*60}")
    print(f"  AutoQuant complete after {iteration - 1} iterations")
    print(f"  Keep: {total_keep}, Discard: {total_discard}, Crash: {total_crash}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
