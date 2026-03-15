"""
Analysis and reporting for NASDAQ quantitative trading system.
Generates IC analysis, portfolio performance reports, and visualizations.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal Analysis (IC)
# ---------------------------------------------------------------------------

def compute_ic_series(
    pred_scores: pd.Series,  # MultiIndex (date, symbol)
    actual_returns: pd.Series,  # MultiIndex (date, symbol)
) -> pd.DataFrame:
    """
    Compute daily IC (Information Coefficient) between prediction and actual returns.

    Returns:
        DataFrame with columns: date, ic, rank_ic
    """
    # Align on common index
    common_idx = pred_scores.index.intersection(actual_returns.index)
    pred = pred_scores.loc[common_idx]
    actual = actual_returns.loc[common_idx]

    # Group by date
    dates = pred.index.get_level_values("date").unique().sort_values()

    ic_records = []
    for date in dates:
        try:
            p = pred.xs(date, level="date")
            a = actual.xs(date, level="date")
            common = p.index.intersection(a.index)
            if len(common) < 5:
                continue
            p_vals = p[common].values
            a_vals = a[common].values

            # Pearson IC
            ic = np.corrcoef(p_vals, a_vals)[0, 1]
            # Rank IC (Spearman)
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(p_vals, a_vals)

            ic_records.append({
                "date": date,
                "ic": ic,
                "rank_ic": rank_ic,
            })
        except Exception:
            continue

    return pd.DataFrame(ic_records)


def ic_summary(ic_df: pd.DataFrame) -> Dict:
    """Summarize IC statistics."""
    if ic_df.empty:
        return {}

    return {
        "mean_ic": ic_df["ic"].mean(),
        "std_ic": ic_df["ic"].std(),
        "icir": ic_df["ic"].mean() / (ic_df["ic"].std() + 1e-12),
        "mean_rank_ic": ic_df["rank_ic"].mean(),
        "std_rank_ic": ic_df["rank_ic"].std(),
        "rank_icir": ic_df["rank_ic"].mean() / (ic_df["rank_ic"].std() + 1e-12),
        "ic_positive_ratio": (ic_df["ic"] > 0).mean(),
        "rank_ic_positive_ratio": (ic_df["rank_ic"] > 0).mean(),
    }


# ---------------------------------------------------------------------------
# Group Return Analysis
# ---------------------------------------------------------------------------

def group_return_analysis(
    pred_scores: pd.Series,
    actual_returns: pd.Series,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    Analyze returns by prediction score groups (quantiles).

    Group 1 = highest predicted, Group N = lowest predicted.
    """
    common_idx = pred_scores.index.intersection(actual_returns.index)
    pred = pred_scores.loc[common_idx]
    actual = actual_returns.loc[common_idx]

    dates = pred.index.get_level_values("date").unique().sort_values()

    group_returns = {i: [] for i in range(1, n_groups + 1)}

    for date in dates:
        try:
            p = pred.xs(date, level="date")
            a = actual.xs(date, level="date")
            common = p.index.intersection(a.index)
            if len(common) < n_groups * 2:
                continue

            p_vals = p[common]
            a_vals = a[common]

            # Assign groups (1=best, N=worst)
            groups = pd.qcut(p_vals.rank(method="first"), n_groups, labels=False) + 1

            for g in range(1, n_groups + 1):
                mask = groups == g
                if mask.sum() > 0:
                    group_returns[g].append(a_vals[mask].mean())
        except Exception:
            continue

    summary = []
    for g in range(1, n_groups + 1):
        rets = group_returns[g]
        if rets:
            arr = np.array(rets)
            summary.append({
                "group": g,
                "mean_return": arr.mean(),
                "annualized_return": arr.mean() * 252,
                "std": arr.std(),
                "sharpe": arr.mean() / (arr.std() + 1e-12) * np.sqrt(252),
                "n_days": len(rets),
            })

    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    backtest_result: Dict,
    ic_df: Optional[pd.DataFrame] = None,
    group_df: Optional[pd.DataFrame] = None,
    output_dir: str = "quant_results/reports",
) -> str:
    """
    Generate a comprehensive text report.

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("    NASDAQ Quantitative Trading System - Performance Report")
    lines.append("=" * 70)
    lines.append("")

    # --- Portfolio Metrics ---
    metrics = backtest_result.get("metrics", {})
    if metrics:
        lines.append("--- Portfolio Performance ---")
        lines.append(f"  Total Return:          {metrics.get('total_return', 0)*100:.2f}%")
        lines.append(f"  Annualized Return:     {metrics.get('annualized_return', 0)*100:.2f}%")
        lines.append(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%")
        lines.append(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  Max Drawdown:          {metrics.get('max_drawdown', 0)*100:.2f}%")
        lines.append(f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0):.4f}")
        lines.append(f"  Win Rate:              {metrics.get('win_rate', 0)*100:.2f}%")
        lines.append(f"  Avg Daily Return:      {metrics.get('avg_daily_return', 0)*100:.4f}%")
        lines.append(f"  Avg Turnover:          {metrics.get('avg_turnover', 0)*100:.2f}%")
        lines.append(f"  Total Cost:            ${metrics.get('total_cost', 0)*metrics.get('final_equity', 1e6):.2f}")
        lines.append(f"  Profit/Loss Ratio:     {metrics.get('profit_loss_ratio', 0):.4f}")
        lines.append(f"  Trading Days:          {metrics.get('n_trading_days', 0)}")
        lines.append(f"  Final Equity:          ${metrics.get('final_equity', 0):,.2f}")
        lines.append("")

    # --- IC Analysis ---
    if ic_df is not None and not ic_df.empty:
        ic_stats = ic_summary(ic_df)
        lines.append("--- Signal (IC) Analysis ---")
        lines.append(f"  Mean IC:               {ic_stats.get('mean_ic', 0):.4f}")
        lines.append(f"  IC Std:                {ic_stats.get('std_ic', 0):.4f}")
        lines.append(f"  ICIR:                  {ic_stats.get('icir', 0):.4f}")
        lines.append(f"  Mean Rank IC:          {ic_stats.get('mean_rank_ic', 0):.4f}")
        lines.append(f"  Rank ICIR:             {ic_stats.get('rank_icir', 0):.4f}")
        lines.append(f"  IC > 0 Ratio:          {ic_stats.get('ic_positive_ratio', 0)*100:.1f}%")
        lines.append(f"  Rank IC > 0 Ratio:     {ic_stats.get('rank_ic_positive_ratio', 0)*100:.1f}%")
        lines.append("")

    # --- Group Analysis ---
    if group_df is not None and not group_df.empty:
        lines.append("--- Group Return Analysis ---")
        lines.append(f"  {'Group':>6} {'Mean Ret':>12} {'Ann Ret':>12} {'Sharpe':>10}")
        lines.append(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
        for _, row in group_df.iterrows():
            g = int(row["group"])
            mr = row["mean_return"] * 100
            ar = row["annualized_return"] * 100
            sr = row["sharpe"]
            label = "Top" if g == 1 else ("Bottom" if g == group_df["group"].max() else f"G{g}")
            lines.append(f"  {label:>6} {mr:>11.4f}% {ar:>11.2f}% {sr:>10.4f}")
        lines.append("")

        # Long-short spread
        if len(group_df) >= 2:
            top_ret = group_df.iloc[0]["annualized_return"]
            bot_ret = group_df.iloc[-1]["annualized_return"]
            spread = (top_ret - bot_ret) * 100
            lines.append(f"  Long-Short Spread:   {spread:.2f}% annualized")
            lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    # Save report
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "performance_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(
    backtest_result: Dict,
    ic_df: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: str = "quant_results/reports",
) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available - skipping plots")
        return

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    portfolio = backtest_result.get("portfolio")
    if portfolio is None or portfolio.empty:
        print("No portfolio data to plot.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("NASDAQ Quantitative Trading System - Analysis", fontsize=16, fontweight="bold")

    # 1. Cumulative return
    ax = axes[0, 0]
    ax.plot(portfolio.index, portfolio["cum_return"], label="Strategy", color="blue", linewidth=1.5)
    if benchmark_returns is not None and not benchmark_returns.empty:
        common_dates = portfolio.index.intersection(benchmark_returns.index)
        if len(common_dates) > 0:
            bench_cum = (1 + benchmark_returns.loc[common_dates]).cumprod()
            ax.plot(common_dates, bench_cum, label="NASDAQ 100", color="gray", linewidth=1, alpha=0.7)
    ax.set_title("Cumulative Return")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    cum_ret = portfolio["cum_return"]
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret - peak) / peak
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    # 3. Daily returns distribution
    ax = axes[1, 0]
    ax.hist(portfolio["return"], bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    # 4. Rolling Sharpe
    ax = axes[1, 1]
    rolling_return = portfolio["return"].rolling(60).mean()
    rolling_std = portfolio["return"].rolling(60).std()
    rolling_sharpe = (rolling_return / (rolling_std + 1e-12)) * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="green", linewidth=1)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Rolling 60-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    # 5. IC time series
    ax = axes[2, 0]
    if ic_df is not None and not ic_df.empty:
        ic_dates = pd.to_datetime(ic_df["date"])
        ax.bar(ic_dates, ic_df["ic"], color="steelblue", alpha=0.6, width=1)
        ax.axhline(y=ic_df["ic"].mean(), color="red", linestyle="--",
                    label=f"Mean IC: {ic_df['ic'].mean():.4f}")
        ax.set_title("Daily IC")
        ax.set_ylabel("IC")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No IC data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Daily IC")
    ax.grid(True, alpha=0.3)

    # 6. Turnover
    ax = axes[2, 1]
    ax.bar(portfolio.index, portfolio["turnover"], color="orange", alpha=0.6, width=1)
    ax.axhline(y=portfolio["turnover"].mean(), color="red", linestyle="--",
               label=f"Avg: {portfolio['turnover'].mean()*100:.1f}%")
    ax.set_title("Daily Turnover")
    ax.set_ylabel("Turnover")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "analysis_report.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis plots saved to {plot_path}")


def plot_equity_curve(
    backtest_result: Dict,
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: str = "quant_results/reports",
) -> None:
    """Plot a clean equity curve chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    portfolio = backtest_result.get("portfolio")
    if portfolio is None or portfolio.empty:
        return

    metrics = backtest_result.get("metrics", {})

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(portfolio.index, portfolio["equity"], label="Strategy", color="blue", linewidth=1.5)

    if benchmark_returns is not None and not benchmark_returns.empty:
        common_dates = portfolio.index.intersection(benchmark_returns.index)
        if len(common_dates) > 0:
            bench_cum = (1 + benchmark_returns.loc[common_dates]).cumprod()
            bench_equity = 1_000_000 * bench_cum
            ax.plot(common_dates, bench_equity, label="NASDAQ 100 (Buy & Hold)",
                    color="gray", linewidth=1, alpha=0.7)

    ax.set_title("NASDAQ Quant Strategy - Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add metrics text
    txt = (
        f"Ann. Return: {metrics.get('annualized_return', 0)*100:.1f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Max DD: {metrics.get('max_drawdown', 0)*100:.1f}%"
    )
    ax.text(0.5, 0.02, txt, transform=ax.transAxes, fontsize=10,
            ha="center", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plot_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Equity curve saved to {plot_path}")
