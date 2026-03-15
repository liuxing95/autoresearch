#!/usr/bin/env python3
"""
AutoQuant v2: Robust Autonomous Experiment Loop

Key improvements over v1:
  1. Rolling walk-forward training (retrain periodically with recent data)
  2. Robustness scoring (IC + long-short + Sharpe + drawdown composite)
  3. Multi-period evaluation (must perform on BOTH 2023-2024 AND 2025)
  4. Expanded search space (train window, feature selection, etc.)
  5. IC-based feature selection to reduce overfitting

Usage:
    python autorun.py --max-iter 20       # Run 20 experiments
    python autorun.py --symbols AAPL,MSFT,GOOGL   # Custom symbols
    python autorun.py --mode robust       # Multi-period robustness mode (default)
    python autorun.py --mode sharpe       # Legacy sharpe-only mode
"""

import os
import sys
import time
import random
import argparse
import subprocess
import datetime
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_TSV = os.path.join("quant_results", "autorun_results.tsv")


# ---------------------------------------------------------------------------
# Experiment parameter space (v2 — much richer)
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    # Model
    "model": ["lightgbm", "linear"],

    # Strategy
    "topk": [5, 10, 15, 20, 30],
    "n_drop": [1, 2, 3, 5],
    "rebalance_freq": ["day", "week"],

    # LightGBM hyperparams
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
    "max_depth": [4, 5, 6, 7, 8],
    "num_leaves": [31, 48, 64, 96, 128],
    "lambda_l1": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "lambda_l2": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],

    # Risk controls
    "stop_loss": [-0.05, -0.08, -0.10, -0.15, -1.0],
    "max_drawdown_limit": [-0.15, -0.20, -0.25, -0.30, -0.50],

    # Rolling training (NEW — critical for adaptation)
    "rolling": [True, False],
    "train_window": [2, 3, 4, 5],
    "retrain_freq": [3, 6, 12],

    # Feature selection (NEW — reduces overfitting)
    "feature_select": [True, False],
    "n_features": [30, 50, 80, 100, 142],
}


def load_results() -> list:
    """Load previous experiment results."""
    results = []
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 9:
                    results.append({
                        "timestamp": parts[0],
                        "sharpe": float(parts[1]),
                        "max_dd": float(parts[2]),
                        "annual_ret": float(parts[3]),
                        "robustness": float(parts[4]),
                        "status": parts[5],
                        "model": parts[6],
                        "mode": parts[7],
                        "description": parts[8],
                    })
                elif len(parts) >= 7:
                    # Legacy format compatibility
                    results.append({
                        "timestamp": parts[0],
                        "sharpe": float(parts[1]),
                        "max_dd": float(parts[2]),
                        "annual_ret": float(parts[3]),
                        "robustness": float(parts[1]),
                        "status": parts[4],
                        "model": parts[5],
                        "mode": "sharpe",
                        "description": parts[6],
                    })
    return results


def get_best_score(results: list, mode: str = "robust") -> float:
    """Get the best score from kept experiments."""
    kept = [r for r in results if r["status"] == "keep"]
    if not kept:
        return -999.0
    if mode == "robust":
        return max(r["robustness"] for r in kept)
    return max(r["sharpe"] for r in kept)


def mutate_params(results: list) -> dict:
    """Generate a new set of experiment parameters by mutation."""
    params = {
        "model": random.choice(SEARCH_SPACE["model"]),
        "topk": random.choice(SEARCH_SPACE["topk"]),
        "n_drop": random.choice(SEARCH_SPACE["n_drop"]),
        "rebalance_freq": random.choice(SEARCH_SPACE["rebalance_freq"]),
        "stop_loss": random.choice(SEARCH_SPACE["stop_loss"]),
        "max_drawdown_limit": random.choice(SEARCH_SPACE["max_drawdown_limit"]),
        "rolling": random.choice(SEARCH_SPACE["rolling"]),
        "feature_select": random.choice(SEARCH_SPACE["feature_select"]),
    }

    # Rolling training params
    if params["rolling"]:
        params["train_window"] = random.choice(SEARCH_SPACE["train_window"])
        params["retrain_freq"] = random.choice(SEARCH_SPACE["retrain_freq"])

    # Feature selection
    if params["feature_select"]:
        params["n_features"] = random.choice(SEARCH_SPACE["n_features"])

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


def _build_cmd(params: dict, symbols: str, test_start: str, test_end: str) -> list:
    """Build run_pipeline.py command for a given test period."""
    cmd = [
        sys.executable, "run_pipeline.py",
        "--step", "all",
        "--model", params["model"],
        "--topk", str(params["topk"]),
        "--n-drop", str(params["n_drop"]),
        "--stop-loss", str(params["stop_loss"]),
        "--max-drawdown-limit", str(params["max_drawdown_limit"]),
        "--rebalance-freq", params["rebalance_freq"],
        "--test-start", test_start,
        "--test-end", test_end,
    ]

    if params["model"] == "lightgbm":
        cmd.extend(["--learning-rate", str(params["learning_rate"])])
        cmd.extend(["--max-depth", str(params["max_depth"])])
        cmd.extend(["--num-leaves", str(params["num_leaves"])])
        cmd.extend(["--lambda-l1", str(params["lambda_l1"])])
        cmd.extend(["--lambda-l2", str(params["lambda_l2"])])
        cmd.extend(["--subsample", str(params.get("subsample", 0.8))])
        cmd.extend(["--colsample-bytree", str(params.get("colsample_bytree", 0.8))])

    if params.get("rolling"):
        cmd.append("--rolling")
        cmd.extend(["--train-window", str(params.get("train_window", 3))])
        cmd.extend(["--retrain-freq", str(params.get("retrain_freq", 6))])

    if params.get("feature_select"):
        cmd.append("--feature-select")
        cmd.extend(["--n-features", str(params.get("n_features", 80))])

    if symbols:
        cmd.extend(["--symbols", symbols])

    return cmd


def _run_single_period(params: dict, symbols: str,
                       test_start: str, test_end: str,
                       period_name: str) -> dict:
    """Run a single backtest period and extract metrics."""
    cmd = _build_cmd(params, symbols, test_start, test_end)

    print(f"\n  [{period_name}] Running {test_start} ~ {test_end}")
    print(f"  Command: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
        )
        output = result.stdout + result.stderr

        sharpe = _extract_metric(output, "sharpe_ratio")
        max_dd = _extract_metric(output, "max_drawdown")
        annual_ret = _extract_metric(output, "annualized_return")
        ic = _extract_metric(output, "Mean IC")
        icir = _extract_metric(output, "ICIR")
        ls_spread = _extract_metric(output, "Long-Short Spread")
        circuit_days = _extract_metric(output, "circuit_breaker_days")
        stop_events = _extract_metric(output, "stop_loss_events")

        if sharpe is None:
            return {"status": "crash", "sharpe": 0, "max_dd": 0,
                    "annual_ret": 0, "ic": 0, "icir": 0, "ls_spread": 0,
                    "circuit_days": 0}

        return {
            "status": "ok",
            "sharpe": sharpe or 0,
            "max_dd": max_dd or 0,
            "annual_ret": annual_ret or 0,
            "ic": ic or 0,
            "icir": icir or 0,
            "ls_spread": ls_spread or 0,
            "circuit_days": circuit_days or 0,
            "stop_events": stop_events or 0,
        }

    except subprocess.TimeoutExpired:
        return {"status": "crash", "sharpe": 0, "max_dd": 0,
                "annual_ret": 0, "ic": 0, "icir": 0, "ls_spread": 0,
                "circuit_days": 0}
    except Exception as e:
        print(f"  Experiment failed: {e}")
        return {"status": "crash", "sharpe": 0, "max_dd": 0,
                "annual_ret": 0, "ic": 0, "icir": 0, "ls_spread": 0,
                "circuit_days": 0}


def run_experiment_robust(params: dict, symbols: str = None) -> dict:
    """
    Run experiment with multi-period evaluation.

    Tests on:
    1. 2023-2024 (in-sample/recent past)
    2. 2025-01-01 to 2025-12-31 (out-of-sample/current)

    Computes a robustness score that rewards consistency across periods.
    """
    print(f"\n{'='*60}")
    print(f"  Multi-Period Experiment: {params['model']} "
          f"topk={params['topk']} "
          f"rolling={params.get('rolling', False)}")
    print(f"{'='*60}")

    # Period 1: 2023-2024
    p1 = _run_single_period(params, symbols, "2023-01-01", "2024-12-31", "Period1 2023-2024")

    if p1["status"] == "crash":
        return {"sharpe": 0, "max_dd": 0, "annual_ret": 0, "robustness": -999,
                "status": "crash", "detail": "P1 crash"}

    # Period 2: 2025
    p2 = _run_single_period(params, symbols, "2025-01-01", "2025-12-31", "Period2 2025")

    if p2["status"] == "crash":
        return {"sharpe": p1["sharpe"], "max_dd": p1["max_dd"],
                "annual_ret": p1["annual_ret"], "robustness": -999,
                "status": "crash", "detail": "P2 crash"}

    # Compute robustness score
    robustness = compute_robustness_score(p1, p2)

    # Use combined metrics
    combined_sharpe = (p1["sharpe"] + p2["sharpe"]) / 2
    combined_dd = min(p1["max_dd"], p2["max_dd"])
    combined_ret = (p1["annual_ret"] + p2["annual_ret"]) / 2

    # Hard constraints
    if combined_dd < -0.35:
        return {"sharpe": combined_sharpe, "max_dd": combined_dd,
                "annual_ret": combined_ret, "robustness": robustness,
                "status": "crash", "detail": f"DD too deep: {combined_dd:.2%}"}

    return {
        "sharpe": combined_sharpe,
        "max_dd": combined_dd,
        "annual_ret": combined_ret,
        "robustness": robustness,
        "status": "pending",
        "p1_sharpe": p1["sharpe"],
        "p2_sharpe": p2["sharpe"],
        "p1_ic": p1["ic"],
        "p2_ic": p2["ic"],
        "detail": f"P1:{p1['sharpe']:.2f}/{p1['ic']:.3f} P2:{p2['sharpe']:.2f}/{p2['ic']:.3f}",
    }


def run_experiment_sharpe(params: dict, symbols: str = None) -> dict:
    """Legacy mode: single period, sharpe-only evaluation."""
    cmd = _build_cmd(params, symbols, "2023-01-01", "2025-12-31")

    print(f"\n{'='*60}")
    print(f"  Sharpe-Only Experiment: {params['model']} topk={params['topk']}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr

        sharpe = _extract_metric(output, "sharpe_ratio")
        max_dd = _extract_metric(output, "max_drawdown")
        annual_ret = _extract_metric(output, "annualized_return")

        if sharpe is None:
            return {"sharpe": 0, "max_dd": 0, "annual_ret": 0,
                    "robustness": 0, "status": "crash"}

        if max_dd is not None and max_dd < -0.35:
            return {"sharpe": sharpe, "max_dd": max_dd,
                    "annual_ret": annual_ret or 0,
                    "robustness": sharpe, "status": "crash"}

        return {
            "sharpe": sharpe,
            "max_dd": max_dd or 0,
            "annual_ret": annual_ret or 0,
            "robustness": sharpe,
            "status": "pending",
        }

    except subprocess.TimeoutExpired:
        return {"sharpe": 0, "max_dd": 0, "annual_ret": 0,
                "robustness": 0, "status": "crash"}
    except Exception as e:
        print(f"  Experiment failed: {e}")
        return {"sharpe": 0, "max_dd": 0, "annual_ret": 0,
                "robustness": 0, "status": "crash"}


def compute_robustness_score(p1: dict, p2: dict) -> float:
    """
    Compute a composite robustness score from two evaluation periods.

    Components:
    1. Combined Sharpe (capped at 3.0 to avoid chasing outliers)
    2. IC consistency (positive IC in both periods)
    3. Drawdown penalty
    4. Period consistency (both periods should be positive)
    5. Circuit breaker penalty (less is better)

    Score range: roughly -2 to +3, higher is better.
    """
    score = 0.0

    # Component 1: Combined Sharpe (weight 0.35)
    s1 = min(max(p1["sharpe"], -2), 3.0)
    s2 = min(max(p2["sharpe"], -2), 3.0)
    sharpe_component = (0.4 * s1 + 0.6 * s2)
    score += 0.35 * sharpe_component

    # Component 2: IC quality (weight 0.25)
    ic_component = 0.0
    if p1["ic"] > 0:
        ic_component += min(p1["ic"] / 0.03, 1.0) * 0.4
    else:
        ic_component -= 0.3
    if p2["ic"] > 0:
        ic_component += min(p2["ic"] / 0.03, 1.0) * 0.6
    else:
        ic_component -= 0.5
    score += 0.25 * ic_component

    # Component 3: Drawdown penalty (weight 0.20)
    dd1 = max(p1["max_dd"], -0.5)
    dd2 = max(p2["max_dd"], -0.5)
    dd_component = 1.0 + (dd1 + dd2)
    score += 0.20 * dd_component

    # Component 4: Consistency across periods (weight 0.15)
    consistency = 0.0
    if p1["sharpe"] > 0 and p2["sharpe"] > 0:
        consistency = 1.0
    elif p1["sharpe"] > 0 or p2["sharpe"] > 0:
        consistency = 0.3
    else:
        consistency = -0.5
    score += 0.15 * consistency

    # Component 5: Circuit breaker penalty (weight 0.05)
    cb_penalty = 0.0
    total_days = 499 + 247
    cb_days = p1.get("circuit_days", 0) + p2.get("circuit_days", 0)
    if cb_days > 0:
        cb_ratio = cb_days / total_days
        cb_penalty = -cb_ratio * 2
    score += 0.05 * cb_penalty

    return round(score, 4)


def _extract_metric(output: str, metric_name: str) -> float:
    """Extract a metric value from pipeline output."""
    pattern = rf"{re.escape(metric_name)}\s*:\s*([-+]?\d+\.?\d*)(%?)"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        has_pct = match.group(2) == "%"
        if has_pct:
            val = val / 100.0
        return val
    return None


def log_result(params: dict, result: dict, mode: str) -> None:
    """Log experiment result to TSV file."""
    os.makedirs(os.path.dirname(RESULTS_TSV), exist_ok=True)

    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("timestamp\tsharpe\tmax_dd\tannual_ret\trobustness\tstatus\tmodel\tmode\tdescription\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    desc = _make_description(params, result)

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{ts}\t{result['sharpe']:.4f}\t{result['max_dd']:.4f}\t"
                f"{result['annual_ret']:.4f}\t{result.get('robustness', 0):.4f}\t"
                f"{result['status']}\t{params['model']}\t{mode}\t{desc}\n")


def _make_description(params: dict, result: dict) -> str:
    """Create a short description of the experiment."""
    parts = [f"{params['model']}"]
    parts.append(f"topk={params['topk']}")
    parts.append(f"n_drop={params['n_drop']}")
    parts.append(f"freq={params['rebalance_freq']}")

    if params.get("rolling"):
        parts.append(f"roll={params.get('train_window', 3)}yr/{params.get('retrain_freq', 6)}mo")
    if params.get("feature_select"):
        parts.append(f"feat={params.get('n_features', 80)}")
    if params["model"] == "lightgbm":
        parts.append(f"lr={params.get('learning_rate', '?')}")
        parts.append(f"d={params.get('max_depth', '?')}")
        parts.append(f"l1={params.get('lambda_l1', '?')}")
    if params.get("stop_loss", -1.0) > -1.0:
        parts.append(f"sl={params['stop_loss']}")
    if result.get("detail"):
        parts.append(f"[{result['detail']}]")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AutoQuant v2: Robust Autonomous Experiment Loop")
    parser.add_argument("--max-iter", type=int, default=0,
                        help="Max iterations (0 = infinite)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated stock symbols")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--mode", type=str, default="robust",
                        choices=["robust", "sharpe"],
                        help="Evaluation mode: robust (multi-period) or sharpe (legacy)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("  AutoQuant v2: Robust Autonomous Experiment Loop")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Max iterations: {'infinite' if args.max_iter == 0 else args.max_iter}")
    print(f"  Results file:   {RESULTS_TSV}")
    print(f"  Symbols:        {args.symbols or 'NASDAQ 100 (default)'}")
    if args.mode == "robust":
        print(f"  Periods:        2023-2024 + 2025 (multi-period)")
        print(f"  Score:          composite (sharpe + IC + consistency + DD)")
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
        best_score = get_best_score(results, args.mode)
        print(f"  Historical experiments: {len(results)}")
        print(f"  Best {'robustness' if args.mode == 'robust' else 'sharpe'} so far: {best_score:.4f}")

        # Mutate parameters
        params = mutate_params(results)
        print(f"  New params: model={params['model']} topk={params['topk']} "
              f"rolling={params.get('rolling')} feat_sel={params.get('feature_select')}")

        # Run experiment
        t0 = time.time()
        if args.mode == "robust":
            exp_result = run_experiment_robust(params, symbols=args.symbols)
        else:
            exp_result = run_experiment_sharpe(params, symbols=args.symbols)
        dt = time.time() - t0

        # Decide: keep or discard
        score_key = "robustness" if args.mode == "robust" else "sharpe"
        current_score = exp_result.get(score_key, exp_result.get("sharpe", 0))

        if exp_result["status"] == "crash":
            total_crash += 1
            print(f"\n  CRASH — {exp_result.get('detail', 'experiment failed')}")
        elif current_score > best_score:
            exp_result["status"] = "keep"
            total_keep += 1
            print(f"\n  KEEP — {score_key} {current_score:.4f} > {best_score:.4f}")
        else:
            exp_result["status"] = "discard"
            total_discard += 1
            print(f"\n  DISCARD — {score_key} {current_score:.4f} <= {best_score:.4f}")

        # Log
        log_result(params, exp_result, args.mode)

        print(f"  Time: {dt:.1f}s")
        print(f"  Sharpe: {exp_result['sharpe']:.4f}, MaxDD: {exp_result['max_dd']:.4f}, "
              f"AnnRet: {exp_result['annual_ret']:.4f}")
        if args.mode == "robust":
            print(f"  Robustness: {exp_result.get('robustness', 0):.4f}")
            if exp_result.get("detail"):
                print(f"  Detail: {exp_result['detail']}")
        print(f"  Totals — Keep: {total_keep}, Discard: {total_discard}, Crash: {total_crash}")

    print(f"\n{'='*60}")
    print(f"  AutoQuant v2 complete after {iteration - 1} iterations")
    print(f"  Keep: {total_keep}, Discard: {total_discard}, Crash: {total_crash}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
