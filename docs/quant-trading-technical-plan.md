# 量化交易系统技术方案

> 将 `autoresearch` 自主迭代能力与 `ai-hedge-fund` 多 Agent 投资决策框架深度融合，构建**可持续进化的 AI 量化交易平台**。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI Quant Platform                            │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────────┐  │
│  │  Data Layer  │───▶│  Strategy Research │───▶│  Execution     │  │
│  │  (行情/基本面)│    │  (autoresearch风格)│    │  Layer         │  │
│  └──────────────┘    └────────────────────┘    └────────────────┘  │
│         │                      │                       │            │
│         ▼                      ▼                       ▼            │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────────┐  │
│  │  Market Data │    │  AI Agent Ensemble │    │  Risk Control  │  │
│  │  Store       │    │  (ai-hedge-fund)   │    │  & Monitor     │  │
│  └──────────────┘    └────────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 仓库结构规划

建议在 fork 的基础上，按以下目录组织代码：

```
ai-quant-platform/
├── data/                      # 数据层
│   ├── providers/
│   │   ├── yfinance_provider.py    # 美股
│   │   ├── tushare_provider.py     # A 股
│   │   └── akshare_provider.py     # 港股 / A 股备用
│   └── store.py               # 统一数据存取接口
│
├── strategy/                  # 策略层（agent 自主迭代的核心文件）
│   ├── base_strategy.py       # 抽象基类
│   ├── momentum.py            # 策略示例：动量
│   ├── mean_reversion.py      # 策略示例：均值回归
│   └── strategy_template.py  # agent 迭代时修改此文件（类比 train.py）
│
├── agents/                    # AI Agent 层（从 ai-hedge-fund 迁移 + 扩展）
│   ├── base_agent.py
│   ├── fundamental_agent.py
│   ├── technical_agent.py
│   ├── sentiment_agent.py
│   ├── buffett_agent.py
│   └── ensemble.py            # 多路 agent 投票汇总
│
├── backtest/                  # 回测引擎
│   ├── engine.py              # 回测主逻辑
│   ├── metrics.py             # Sharpe / MaxDD / CAGR / Calmar
│   └── walk_forward.py        # Walk-Forward 优化
│
├── risk/                      # 风控层
│   ├── position_sizer.py      # 仓位计算（Kelly / 等权 / 风险平价）
│   └── circuit_breaker.py     # 熔断：最大回撤 / 单笔止损
│
├── execution/                 # 执行层
│   ├── paper_broker.py        # 纸面交易模拟
│   ├── alpaca_broker.py       # Alpaca 实盘
│   └── broker_interface.py    # 统一接口
│
├── research/                  # 自主迭代研究（autoresearch 核心范式）
│   ├── experiment_loop.py     # 主迭代循环（类比 autoresearch program.md 驱动）
│   ├── evaluator.py           # 指标评估
│   └── results.tsv            # 实验记录（不提交 git）
│
├── dashboard/                 # 可视化
│   └── app.py                 # Streamlit 应用
│
├── program.md                 # agent 迭代指令（从 autoresearch 迁移）
├── .env.example
└── README.md
```

---

## 3. 核心模块详解

### 3.1 数据层

```python
# data/store.py
from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """返回 OHLCV DataFrame，index 为 DatetimeIndex"""
        ...

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> dict:
        """返回基本面数据字典"""
        ...
```

```python
# data/providers/yfinance_provider.py
import yfinance as yf
from data.store import MarketDataProvider
import pandas as pd

class YFinanceProvider(MarketDataProvider):
    def get_price_history(self, symbol, start, end, interval="1d"):
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_fundamentals(self, symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "revenue_growth": info.get("revenueGrowth"),
        }
```

### 3.2 策略层 — agent 迭代的核心文件

`strategy/strategy_template.py` 是 agent 唯一可以修改的文件（类比 autoresearch 的 `train.py`）：

```python
# strategy/strategy_template.py
# =============================================
# 此文件由 AI agent 自主迭代，人类不直接修改
# =============================================
import pandas as pd
import numpy as np

# ── 超参数（agent 调整这里）──────────────────
LOOKBACK_SHORT = 20    # 短期均线周期
LOOKBACK_LONG  = 60    # 长期均线周期
RSI_PERIOD     = 14
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
STOP_LOSS_PCT  = 0.05  # 5% 止损
# ─────────────────────────────────────────────

def generate_signals(prices: pd.DataFrame) -> pd.Series:
    """
    输入: OHLCV DataFrame
    输出: signal Series，值为 1（买）/ -1（卖）/ 0（持仓不变）
    """
    close = prices["Close"]

    # 双均线
    ma_short = close.rolling(LOOKBACK_SHORT).mean()
    ma_long  = close.rolling(LOOKBACK_LONG).mean()
    cross_up   = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    cross_down = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    # RSI 过滤
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

    signal = pd.Series(0, index=close.index)
    signal[cross_up  & (rsi < RSI_OVERBOUGHT)] =  1
    signal[cross_down & (rsi > RSI_OVERSOLD)]  = -1
    return signal
```

### 3.3 回测引擎

```python
# backtest/engine.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class BacktestResult:
    sharpe:     float
    max_dd:     float
    cagr:       float
    calmar:     float
    total_ret:  float
    equity:     pd.Series

def run_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> BacktestResult:
    signals = signal_fn(prices)
    close   = prices["Close"]

    position = 0
    cash     = initial_capital
    equity   = []

    for i in range(len(close)):
        sig = signals.iloc[i]
        price = close.iloc[i]

        if sig == 1 and position == 0:
            shares   = cash // price
            cost     = shares * price * (1 + commission)
            cash    -= cost
            position = shares
        elif sig == -1 and position > 0:
            proceeds  = position * price * (1 - commission)
            cash     += proceeds
            position  = 0

        equity.append(cash + position * price)

    equity = pd.Series(equity, index=close.index)

    # 指标计算
    ret         = equity.pct_change().dropna()
    sharpe      = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    n_years     = len(equity) / 252
    cagr        = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    calmar      = cagr / (abs(max_dd) + 1e-9)
    total_ret   = (equity.iloc[-1] / equity.iloc[0]) - 1

    return BacktestResult(
        sharpe=round(sharpe, 4),
        max_dd=round(max_dd, 4),
        cagr=round(cagr, 4),
        calmar=round(calmar, 4),
        total_ret=round(total_ret, 4),
        equity=equity,
    )
```

### 3.4 自主迭代循环（autoresearch 核心范式移植）

```python
# research/experiment_loop.py
"""
AI agent 驱动的策略迭代循环。
类比 autoresearch 的实验循环，但目标指标换成 Sharpe / CAGR。

循环逻辑：
1. 读取当前 git 状态
2. 让 LLM agent 修改 strategy/strategy_template.py
3. git commit
4. 运行回测，获取核心指标
5. 若 Sharpe 提升 → 保留 commit（"advance"）
6. 若 Sharpe 未提升 → git reset 回上一个 commit（"discard"）
7. 记录到 results.tsv
8. 永不停止，直到人类中断
"""
import subprocess
import csv
from pathlib import Path
from backtest.engine import run_backtest
from data.store import MarketDataProvider
import importlib.util

RESULTS_FILE = Path("research/results.tsv")
STRATEGY_FILE = "strategy/strategy_template.py"

def _load_signal_fn():
    spec = importlib.util.spec_from_file_location("strategy_template", STRATEGY_FILE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_signals

def _git_short_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

def _git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)

def record_result(commit, result, status, description):
    header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if header:
            writer.writerow(["commit", "sharpe", "max_dd", "cagr", "status", "description"])
        writer.writerow([
            commit,
            f"{result.sharpe:.6f}",
            f"{result.max_dd:.6f}",
            f"{result.cagr:.6f}",
            status,
            description,
        ])

def experiment_loop(provider: MarketDataProvider, symbols: list[str], **kwargs):
    best_sharpe = float("-inf")

    while True:  # NEVER STOP — 按 Ctrl+C 中断
        # --- agent 修改 strategy_template.py，然后 git commit ---
        # （实际由 LLM agent 完成，此处为框架占位）

        commit = _git_short_hash()
        try:
            signal_fn = _load_signal_fn()
            results = []
            for sym in symbols:
                prices = provider.get_price_history(sym, **kwargs)
                r = run_backtest(prices, signal_fn)
                results.append(r)

            avg_sharpe = sum(r.sharpe for r in results) / len(results)
            avg_cagr   = sum(r.cagr   for r in results) / len(results)

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                status = "keep"
            else:
                status = "discard"
                _git_reset()

            record_result(commit, type("R", (), {
                "sharpe": avg_sharpe,
                "max_dd": min(r.max_dd for r in results),
                "cagr":   avg_cagr,
            })(), status, "auto experiment")

        except Exception as e:
            record_result(commit, type("R", (), {
                "sharpe": 0, "max_dd": 0, "cagr": 0
            })(), "crash", str(e)[:80])
            _git_reset()
        except KeyboardInterrupt:
            print("实验循环被手动中断，退出。")
            break
```

### 3.5 AI Agent 集成（ai-hedge-fund 风格）

```python
# agents/ensemble.py
"""多路 agent 投票，综合输出最终信号"""
from dataclasses import dataclass
from enum import Enum

class Signal(str, Enum):
    BUY    = "buy"
    SELL   = "sell"
    HOLD   = "hold"

@dataclass
class AgentVote:
    agent_name: str
    signal:     Signal
    confidence: float   # 0.0 ~ 1.0
    reasoning:  str

class AgentEnsemble:
    def __init__(self, agents: list):
        self.agents = agents

    def vote(self, symbol: str, data: dict) -> Signal:
        votes: list[AgentVote] = [a.analyze(symbol, data) for a in self.agents]

        buy_score  = sum(v.confidence for v in votes if v.signal == Signal.BUY)
        sell_score = sum(v.confidence for v in votes if v.signal == Signal.SELL)

        if buy_score > sell_score * 1.2:
            return Signal.BUY
        elif sell_score > buy_score * 1.2:
            return Signal.SELL
        return Signal.HOLD
```

---

## 4. 与 ai-hedge-fund 的集成方式

### 步骤一：Fork 并清理

```bash
git clone https://github.com/virattt/ai-hedge-fund.git ai-quant-platform
cd ai-quant-platform
# 移除不需要的演示脚本，保留 agents/ 和 backtester/
```

### 步骤二：统一数据接口

ai-hedge-fund 原本使用 `financial_datasets` 库。将其替换为本项目 `data/providers/` 层，保持 `get_price_history` / `get_fundamentals` 接口一致。

### 步骤三：植入 autoresearch 迭代循环

将 autoresearch 的 `program.md` 哲学移植到量化策略迭代：

| autoresearch 概念 | 量化平台对应 |
|---|---|
| `train.py` | `strategy/strategy_template.py` |
| `val_bpb`（越低越好）| `sharpe`（越高越好）|
| `results.tsv` | `research/results.tsv` |
| 5 分钟训练预算 | N 年回测窗口（固定） |
| git commit/reset | 策略版本管理 |

### 步骤四：风控层嵌入

在 `execution/` 层的每次下单前，调用 `risk/circuit_breaker.py` 校验：
- 单笔最大亏损 ≤ 总资产 2%
- 单只股票持仓 ≤ 总资产 20%
- 组合最大回撤触发 15% 熔断，暂停所有下单

---

## 5. 技术选型

| 模块 | 技术栈 |
|---|---|
| 数据获取 | yfinance / Tushare Pro / AKShare |
| 回测引擎 | 自研（基于 pandas）/ 可选 VectorBT |
| AI Agent | LangChain + OpenAI GPT-4o / Claude 3.5 Sonnet |
| 本地 LLM（降本）| Ollama + Qwen2.5-72B |
| 可视化 | Streamlit + Plotly |
| 纸面交易 | Alpaca Paper Trading API / 长桥证券 OpenAPI |
| 版本管理 | Git（同 autoresearch 范式）|
| 依赖管理 | uv（同 autoresearch）|
| 配置管理 | python-dotenv + `.env` |
| 日志 | structlog（JSON Lines）|

---

## 6. 部署架构

```
本地开发 / 云服务器（单机）
│
├── Cron / Supervisor ──▶ research/experiment_loop.py  （自主策略迭代，后台常驻）
│
├── Cron（每交易日收盘后）──▶ backtest/engine.py         （每日增量回测）
│
├── Cron（每交易日开盘前）──▶ agents/ensemble.py         （生成当日信号）
│
├── execution/paper_broker.py                            （模拟下单，记录日志）
│
└── Streamlit Dashboard                                  （端口 8501，随时查看）
```

---

## 7. 关键代码依赖关系图

```
program.md (策略迭代指令)
    │
    ▼
research/experiment_loop.py
    │
    ├──▶ strategy/strategy_template.py   (agent 修改)
    │
    ├──▶ backtest/engine.py              (运行回测)
    │       │
    │       └──▶ data/store.py           (获取行情)
    │
    ├──▶ agents/ensemble.py              (AI agent 分析)
    │       │
    │       └──▶ data/store.py           (获取基本面 / 新闻)
    │
    └──▶ research/results.tsv            (记录结果)
```
