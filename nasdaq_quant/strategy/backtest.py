"""
Backtesting engine for NASDAQ quantitative trading strategy.
Implements TopK-Dropout strategy and portfolio simulation.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TopK-Dropout Strategy
# ---------------------------------------------------------------------------

class TopKDropoutStrategy:
    """
    TopK-Dropout stock selection strategy.

    Each day:
    1. Rank all stocks by prediction score
    2. Hold top-K stocks
    3. When rebalancing, drop at most n_drop stocks that fell out of top-K
    4. Buy stocks that newly entered top-K to fill positions
    5. Equal weight allocation
    """

    def __init__(
        self,
        topk: int = 30,
        n_drop: int = 5,
        max_weight: float = 0.10,
        rebalance_freq: str = "day",
    ):
        self.topk = topk
        self.n_drop = n_drop
        self.max_weight = max_weight
        self.rebalance_freq = rebalance_freq  # "day" or "week"

    def _is_rebalance_day(self, date, dates_list, idx):
        """Check if this date is a rebalance day based on frequency."""
        if self.rebalance_freq == "day":
            return True
        elif self.rebalance_freq == "week":
            # Rebalance on the first trading day of each week
            if idx == 0:
                return True
            prev_date = dates_list[idx - 1]
            return date.isocalendar()[1] != prev_date.isocalendar()[1]
        return True

    def generate_signals(
        self,
        pred_scores: pd.Series,  # MultiIndex (date, symbol) -> score
    ) -> pd.DataFrame:
        """
        Generate daily portfolio weights from prediction scores.

        Returns:
            DataFrame with columns: date, symbol, weight, rank
        """
        # Unstack to get (date x symbol) matrix
        score_matrix = pred_scores.unstack(level="symbol")
        dates = score_matrix.index.sort_values()

        all_records = []
        prev_holdings = set()
        dates_list = list(dates)

        for idx, date in enumerate(dates_list):
            day_scores = score_matrix.loc[date].dropna()
            if len(day_scores) < self.topk:
                continue

            # Check if this is a rebalance day
            if not self._is_rebalance_day(date, dates_list, idx) and prev_holdings:
                # Not a rebalance day: keep previous holdings unchanged
                holdings = prev_holdings
            else:
                # Rank stocks
                ranked = day_scores.sort_values(ascending=False)
                top_symbols = set(ranked.index[:self.topk])

                if not prev_holdings:
                    # First day: simply take top-K
                    holdings = top_symbols
                else:
                    # Determine which to drop
                    to_drop = prev_holdings - top_symbols
                    if len(to_drop) > self.n_drop:
                        # Only drop n_drop lowest-ranked from to_drop set
                        drop_scores = day_scores[list(to_drop)]
                        to_drop = set(drop_scores.nsmallest(self.n_drop).index)

                    remaining = prev_holdings - to_drop
                    # Fill up to topK from new candidates
                    candidates = top_symbols - remaining
                    n_to_add = self.topk - len(remaining)
                    cand_scores = day_scores[list(candidates)]
                    to_add = set(cand_scores.nlargest(min(n_to_add, len(cand_scores))).index)

                    holdings = remaining | to_add

            # Equal weight
            n_hold = len(holdings)
            if n_hold == 0:
                continue

            weight = min(1.0 / n_hold, self.max_weight)

            for symbol in holdings:
                rank_val = ranked.index.get_loc(symbol) + 1 if symbol in ranked.index else -1
                all_records.append({
                    "date": date,
                    "symbol": symbol,
                    "weight": weight,
                    "rank": rank_val,
                    "score": day_scores.get(symbol, 0),
                })

            prev_holdings = holdings

        signals = pd.DataFrame(all_records)
        if len(signals) > 0:
            signals["date"] = pd.to_datetime(signals["date"])

        return signals


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Simple vectorized backtesting engine.

    Given daily portfolio weights and actual returns, compute strategy performance.
    """

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        buy_cost: float = 0.001,
        sell_cost: float = 0.001,
        slippage: float = 0.001,
        stop_loss: float = -0.08,
        max_drawdown_limit: float = -0.20,
    ):
        self.init_cash = init_cash
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.max_drawdown_limit = max_drawdown_limit

    def run(
        self,
        signals: pd.DataFrame,  # columns: date, symbol, weight
        returns_data: pd.DataFrame,  # MultiIndex (date, symbol), columns must contain 'daily_return'
    ) -> Dict:
        """
        Run backtest simulation.

        Args:
            signals: strategy signals with weights
            returns_data: actual daily returns

        Returns:
            dict with portfolio and performance data
        """
        dates = sorted(signals["date"].unique())

        # Build daily portfolio returns
        portfolio_returns = []
        turnover_list = []
        prev_weights = {}
        cum_return = 1.0
        peak_return = 1.0
        circuit_breaker_active = False
        stop_loss_events = 0
        circuit_breaker_days = 0

        for date in dates:
            # Check drawdown circuit breaker
            current_drawdown = (cum_return - peak_return) / peak_return if peak_return > 0 else 0
            if self.max_drawdown_limit < 0 and current_drawdown <= self.max_drawdown_limit:
                circuit_breaker_active = True

            if circuit_breaker_active:
                # Circuit breaker: no trading, flat position
                portfolio_returns.append({
                    "date": date,
                    "return": 0.0,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "n_holdings": 0,
                    "circuit_breaker": True,
                })
                circuit_breaker_days += 1
                # Reset circuit breaker after 10 days of cooling
                if circuit_breaker_days >= 10:
                    circuit_breaker_active = False
                    circuit_breaker_days = 0
                    prev_weights = {}
                continue

            day_signals = signals[signals["date"] == date]

            # Current target weights
            curr_weights = {}
            for _, row in day_signals.iterrows():
                curr_weights[row["symbol"]] = row["weight"]

            # Apply stop loss: check individual stock returns
            if self.stop_loss < 0 and prev_weights:
                for symbol in list(curr_weights.keys()):
                    try:
                        if (date, symbol) in returns_data.index:
                            stock_ret = returns_data.loc[(date, symbol), "daily_return"]
                            if stock_ret <= self.stop_loss:
                                # Stop loss triggered: remove from portfolio
                                del curr_weights[symbol]
                                stop_loss_events += 1
                    except (KeyError, TypeError):
                        pass

                # Re-normalize weights after stop loss removal
                if curr_weights:
                    total_w = sum(curr_weights.values())
                    if total_w > 0:
                        curr_weights = {s: w / total_w for s, w in curr_weights.items()}

            # Get actual returns for held stocks
            port_ret = 0.0
            for symbol, weight in curr_weights.items():
                try:
                    if (date, symbol) in returns_data.index:
                        stock_ret = returns_data.loc[(date, symbol), "daily_return"]
                    else:
                        stock_ret = 0.0
                except (KeyError, TypeError):
                    stock_ret = 0.0

                port_ret += weight * stock_ret

            # Calculate turnover (sum of absolute weight changes)
            all_symbols = set(list(curr_weights.keys()) + list(prev_weights.keys()))
            turnover = 0.0
            for s in all_symbols:
                w_new = curr_weights.get(s, 0)
                w_old = prev_weights.get(s, 0)
                turnover += abs(w_new - w_old)

            # Transaction cost + slippage
            cost = turnover * (self.buy_cost + self.sell_cost + self.slippage) / 2
            port_ret -= cost

            # Update cumulative return
            cum_return *= (1 + port_ret)
            peak_return = max(peak_return, cum_return)

            portfolio_returns.append({
                "date": date,
                "return": port_ret,
                "turnover": turnover,
                "cost": cost,
                "n_holdings": len(curr_weights),
                "circuit_breaker": False,
            })

            prev_weights = curr_weights
            turnover_list.append(turnover)

        # Build result DataFrame
        result_df = pd.DataFrame(portfolio_returns)
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df = result_df.set_index("date").sort_index()

        # Cumulative returns
        result_df["cum_return"] = (1 + result_df["return"]).cumprod()
        result_df["equity"] = self.init_cash * result_df["cum_return"]

        # Calculate performance metrics
        metrics = self._calculate_metrics(result_df)
        metrics["stop_loss_events"] = stop_loss_events
        metrics["circuit_breaker_days"] = sum(1 for r in portfolio_returns if r.get("circuit_breaker", False))

        return {
            "portfolio": result_df,
            "metrics": metrics,
            "signals": signals,
        }

    def _calculate_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate performance metrics from portfolio returns."""
        returns = portfolio["return"]
        n_days = len(returns)

        if n_days == 0:
            return {}

        # Annualization factor
        ann_factor = 252

        # Total return
        total_return = portfolio["cum_return"].iloc[-1] - 1

        # Annualized return
        years = n_days / ann_factor
        ann_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

        # Volatility
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(ann_factor)

        # Sharpe ratio (assuming risk-free rate = 4%)
        risk_free = 0.04
        sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum_returns = portfolio["cum_return"]
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / max(n_days, 1)

        # Average turnover
        avg_turnover = portfolio["turnover"].mean()

        # Profit/loss ratio
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profit_loss_ratio = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and losses.mean() != 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "avg_daily_return": returns.mean(),
            "avg_turnover": avg_turnover,
            "total_cost": portfolio["cost"].sum(),
            "profit_loss_ratio": profit_loss_ratio,
            "n_trading_days": n_days,
            "final_equity": portfolio["equity"].iloc[-1],
        }

        return metrics


# ---------------------------------------------------------------------------
# Helper: compute daily returns from raw data
# ---------------------------------------------------------------------------

def compute_daily_returns(
    csv_dir: str,
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Compute daily returns from CSV data.

    Returns:
        DataFrame with MultiIndex (date, symbol), column 'daily_return'
    """
    all_returns = []

    for symbol in symbols:
        path = os.path.join(os.path.expanduser(csv_dir), f"{symbol}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.sort_values("date")

        df["daily_return"] = df["close"].pct_change()
        df["symbol"] = symbol
        df = df[["date", "symbol", "daily_return"]].dropna()

        all_returns.append(df)

    if not all_returns:
        return pd.DataFrame()

    combined = pd.concat(all_returns, ignore_index=True)
    combined = combined.set_index(["date", "symbol"]).sort_index()

    return combined


def compute_benchmark_returns(
    csv_dir: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
) -> pd.Series:
    """Compute benchmark (NASDAQ 100) daily returns."""
    path = os.path.join(os.path.expanduser(csv_dir), "benchmark_NDX.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date")
    df["daily_return"] = df["close"].pct_change()
    df = df.set_index("date")

    return df["daily_return"].dropna()
