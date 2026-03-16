# 量化交易全流程上手教程

> 本教程面向已完成 `autoresearch` 基础实验的用户，指导你从零搭建一套基于 `ai-hedge-fund` + `autoresearch` 范式的 AI 量化交易系统，并实现持续自主迭代。

---

## 前置条件

| 条件 | 说明 |
|---|---|
| Python 3.10+ | 已安装 |
| uv | 已安装（`pip install uv` 或参考官网）|
| Git | 已安装 |
| OpenAI / Anthropic API Key | 用于 AI Agent 分析 |
| （可选）Tushare Token | 访问 A 股数据 |
| （可选）Alpaca API Key | 纸面 / 实盘交易 |

---

## 阶段一：Fork 并搭建项目骨架（预计 1 天）

### 步骤 1.1 Fork ai-hedge-fund

```bash
# 在 GitHub 上先 fork https://github.com/virattt/ai-hedge-fund 到你的账号
# 然后 clone 你 fork 后的仓库
git clone https://github.com/<你的用户名>/ai-hedge-fund.git ai-quant-platform
cd ai-quant-platform
```

### 步骤 1.2 添加 autoresearch 核心文件

```bash
# 将 autoresearch 的迭代范式文件复制进来
cp /path/to/autoresearch/program.md ./strategy_program.md

# 创建项目骨架目录
mkdir -p data/providers strategy backtest risk execution research dashboard
```

### 步骤 1.3 安装依赖

在项目根目录创建或更新 `pyproject.toml`：

```toml
[project]
name = "ai-quant-platform"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "yfinance>=0.2.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "python-dotenv>=1.0.0",
    "streamlit>=1.35.0",
    "plotly>=5.22.0",
    "structlog>=24.0.0",
    "vectorbt>=0.26.0",   # 可选，高性能回测
]
```

```bash
uv sync
```

### 步骤 1.4 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Keys：
```

```ini
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TUSHARE_TOKEN=your_token_here
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # 纸面交易
```

---

## 阶段二：数据层搭建（预计 0.5 天）

### 步骤 2.1 实现 YFinance 数据提供者

新建 `data/providers/yfinance_provider.py`（参考技术方案文档中的代码实现）。

### 步骤 2.2 验证数据获取

```python
# 在项目根目录运行
from data.providers.yfinance_provider import YFinanceProvider

provider = YFinanceProvider()
df = provider.get_price_history("AAPL", start="2022-01-01", end="2024-01-01")
print(df.tail())
print(df.shape)  # 应输出约 (502, 5)
```

### 步骤 2.3（可选）接入 Tushare A 股数据

```python
# data/providers/tushare_provider.py
import tushare as ts
import pandas as pd
from data.store import MarketDataProvider

class TushareProvider(MarketDataProvider):
    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_price_history(self, symbol, start, end, interval="1d"):
        # Tushare symbol 格式：600519.SH（贵州茅台）
        df = self.pro.daily(
            ts_code=symbol,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", "")
        )
        df = df.sort_values("trade_date")
        df.index = pd.to_datetime(df["trade_date"])
        return df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low",  "close": "Close", "vol": "Volume"
        })[["Open", "High", "Low", "Close", "Volume"]]

    def get_fundamentals(self, symbol):
        daily_basic = self.pro.daily_basic(ts_code=symbol, fields="pe,pb,total_mv")
        row = daily_basic.iloc[0]
        return {"pe_ratio": row["pe"], "pb_ratio": row["pb"], "market_cap": row["total_mv"]}
```

---

## 阶段三：回测引擎集成（预计 1 天）

### 步骤 3.1 实现回测引擎

将技术方案中 `backtest/engine.py` 的代码复制到 `backtest/engine.py`。

### 步骤 3.2 第一次回测验证

```python
# scripts/run_backtest.py
from data.providers.yfinance_provider import YFinanceProvider
from backtest.engine import run_backtest
from strategy.strategy_template import generate_signals

provider = YFinanceProvider()
prices = provider.get_price_history("SPY", start="2020-01-01", end="2024-01-01")

result = run_backtest(prices, generate_signals)

print(f"Sharpe Ratio : {result.sharpe:.4f}")
print(f"Max Drawdown : {result.max_dd:.2%}")
print(f"CAGR         : {result.cagr:.2%}")
print(f"Total Return : {result.total_ret:.2%}")
```

```bash
uv run scripts/run_backtest.py
```

**预期输出示例：**
```
Sharpe Ratio : 0.8234
Max Drawdown : -18.32%
CAGR         :  9.41%
Total Return : 42.13%
```

### 步骤 3.3 Walk-Forward 回测（防止过拟合）

```python
# backtest/walk_forward.py
from backtest.engine import run_backtest
import pandas as pd

def walk_forward_test(
    prices: pd.DataFrame,
    signal_fn,
    train_window: int = 252,   # 1 年训练窗口
    test_window: int  = 63,    # 1 季度测试窗口
):
    """
    滚动窗口回测，每次用过去 train_window 天训练（参数优化），
    用接下来 test_window 天测试（out-of-sample 评估）。
    """
    results = []
    n = len(prices)
    start = train_window

    while start + test_window <= n:
        # 测试窗口
        test_prices = prices.iloc[start : start + test_window]
        result = run_backtest(test_prices, signal_fn)
        results.append(result.sharpe)
        start += test_window

    avg_oos_sharpe = sum(results) / len(results)
    print(f"Walk-Forward OOS Sharpe: {avg_oos_sharpe:.4f}  (n={len(results)} folds)")
    return avg_oos_sharpe
```

---

## 阶段四：AI Agent 集成（预计 2 天）

### 步骤 4.1 适配 ai-hedge-fund 的 Agent

ai-hedge-fund 原生包含以下 agent，只需适配数据接口：

```
agents/
├── fundamentals_agent.py   # 基本面分析
├── technicals_agent.py     # 技术指标分析
├── sentiment_agent.py      # 新闻情感分析
├── valuation_agent.py      # 估值分析
└── portfolio_manager.py    # 综合决策
```

修改每个 agent 的数据调用，替换为你的 `data/providers/` 层：

```python
# agents/technicals_agent.py （修改示例）
from data.store import MarketDataProvider

class TechnicalsAgent:
    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def analyze(self, symbol: str, start: str, end: str) -> dict:
        prices = self.provider.get_price_history(symbol, start, end)
        # 计算技术指标...
        close = prices["Close"]
        ma20  = close.rolling(20).mean().iloc[-1]
        ma60  = close.rolling(60).mean().iloc[-1]
        current_price = close.iloc[-1]

        if current_price > ma20 > ma60:
            signal = "bullish"
        elif current_price < ma20 < ma60:
            signal = "bearish"
        else:
            signal = "neutral"

        return {"signal": signal, "confidence": 0.7, "agent": "technicals"}
```

### 步骤 4.2 实现多路 Agent 投票

```python
# agents/ensemble.py （参考技术方案实现）
```

### 步骤 4.3 测试 Agent 流水线

```python
# scripts/run_agents.py
from data.providers.yfinance_provider import YFinanceProvider
from agents.technicals_agent import TechnicalsAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.ensemble import AgentEnsemble

provider = YFinanceProvider()

agents = [
    TechnicalsAgent(provider),
    FundamentalsAgent(provider),
]

ensemble = AgentEnsemble(agents)
signal = ensemble.vote("AAPL", {
    "symbol": "AAPL",
    "start": "2024-01-01",
    "end": "2024-12-31",
})
print(f"综合信号: {signal}")
```

---

## 阶段五：植入 autoresearch 迭代循环（预计 1 天）

这是整个系统的核心——让 AI agent **自主迭代策略**，就像 autoresearch 自主迭代模型代码一样。

### 步骤 5.1 创建策略迭代指令文件

新建 `strategy_program.md`（类比 autoresearch 的 `program.md`）：

````markdown
# Strategy Research Program

## Setup

1. Read `strategy/strategy_template.py` for full context.
2. Read `backtest/engine.py` to understand the evaluation metric (Sharpe Ratio — higher is better).
3. Initialize `research/results.tsv` with header row.

## Experimentation

The only file you modify is `strategy/strategy_template.py`.
Everything is fair game: indicator periods, signal logic, filters, thresholds.

**Goal: maximize out-of-sample Sharpe Ratio.**

Run backtest: `uv run scripts/run_backtest.py`

## Output format

```
sharpe:    1.2340
max_dd:   -0.1523
cagr:      0.1241
```

## The experiment loop

LOOP FOREVER:
1. Tune `strategy/strategy_template.py`
2. git commit
3. Run backtest: `uv run scripts/run_backtest.py > run.log 2>&1`
4. Read results: `grep "^Sharpe\|^Max\|^CAGR" run.log`
5. If Sharpe improved → keep commit ("advance")
6. If Sharpe worse → git reset HEAD~1 ("discard")
7. Record to `research/results.tsv`
8. NEVER STOP
````

### 步骤 5.2 启动自主迭代

```bash
# 创建实验分支（类比 autoresearch 的 autoresearch/mar5 分支）
git checkout -b strategy-research/$(date +%Y%m%d)

# 然后将此仓库交给你的 AI agent（Claude / Codex 等），指向 strategy_program.md：
# "Hi, have a look at strategy_program.md and let's kick off strategy research!"
```

### 步骤 5.3 查看实验记录

```bash
cat research/results.tsv
```

输出示例：
```
commit	sharpe	max_dd	cagr	status	description
a1b2c3d	0.823400	-0.183200	0.094100	keep	baseline MA crossover
b2c3d4e	0.951200	-0.162000	0.112300	keep	add RSI filter
c3d4e5f	0.791000	-0.201000	0.081000	discard	switch to Bollinger Bands
d4e5f6g	1.123400	-0.141000	0.134500	keep	add momentum score filter
```

---

## 阶段六：纸面交易（预计 0.5 天）

### 步骤 6.1 连接 Alpaca Paper Trading

```python
# execution/alpaca_broker.py
import alpaca_trade_api as tradeapi
import os

class AlpacaBroker:
    def __init__(self):
        self.api = tradeapi.REST(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            os.environ["ALPACA_BASE_URL"],
        )

    def submit_order(self, symbol: str, qty: int, side: str):
        """side: 'buy' or 'sell'"""
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        print(f"订单提交: {side} {qty} {symbol} — 订单ID: {order.id}")
        return order

    def get_positions(self):
        return self.api.list_positions()

    def get_account(self):
        return self.api.get_account()
```

### 步骤 6.2 每日自动运行策略

```python
# scripts/daily_run.py
"""每个交易日开盘前运行"""
from data.providers.yfinance_provider import YFinanceProvider
from agents.ensemble import AgentEnsemble
from agents.technicals_agent import TechnicalsAgent
from execution.alpaca_broker import AlpacaBroker
from risk.circuit_breaker import CircuitBreaker
import datetime

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

provider = YFinanceProvider()
broker   = AlpacaBroker()
risk     = CircuitBreaker(max_drawdown=0.15, max_position_pct=0.20)

agents   = [TechnicalsAgent(provider)]
ensemble = AgentEnsemble(agents)

end_date   = datetime.date.today().isoformat()
start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

account  = broker.get_account()
portfolio_value = float(account.portfolio_value)

for sym in SYMBOLS:
    signal = ensemble.vote(sym, {"symbol": sym, "start": start_date, "end": end_date})

    if risk.check(broker.get_positions(), portfolio_value):
        if signal == "buy":
            price = float(broker.api.get_latest_trade(sym).price)
            qty = int(portfolio_value * 0.05 / price)  # 5% 仓位
            if qty > 0:
                broker.submit_order(sym, qty, "buy")
        elif signal == "sell":
            # 获取当前持仓数量再下卖单
            positions = {p.symbol: int(p.qty) for p in broker.get_positions()}
            held_qty = positions.get(sym, 0)
            if held_qty > 0:
                broker.submit_order(sym, held_qty, "sell")  # 清仓
    else:
        print(f"⚠️ 风控熔断，跳过 {sym}")
```

```bash
# 添加到 crontab（每个工作日 9:25 运行，美股开盘前）
# crontab -e
25 9 * * 1-5 cd /path/to/ai-quant-platform && uv run scripts/daily_run.py >> logs/daily.log 2>&1
```

---

## 阶段七：可视化 Dashboard（预计 0.5 天）

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.providers.yfinance_provider import YFinanceProvider
from backtest.engine import run_backtest
from strategy.strategy_template import generate_signals

st.set_page_config(page_title="AI Quant Platform", layout="wide")
st.title("🤖 AI 量化交易平台")

# 侧边栏
symbol     = st.sidebar.text_input("股票代码", value="AAPL")
start_date = st.sidebar.date_input("开始日期", value=pd.to_datetime("2022-01-01"))
end_date   = st.sidebar.date_input("结束日期",  value=pd.to_datetime("2024-01-01"))

if st.sidebar.button("运行回测"):
    with st.spinner("回测中..."):
        provider = YFinanceProvider()
        prices   = provider.get_price_history(
            symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat()
        )
        result = run_backtest(prices, generate_signals)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{result.sharpe:.4f}")
    col2.metric("最大回撤",      f"{result.max_dd:.2%}")
    col3.metric("年化收益 CAGR", f"{result.cagr:.2%}")
    col4.metric("总收益",        f"{result.total_ret:.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity.index,
        y=result.equity.values,
        mode="lines",
        name="策略净值",
        line=dict(color="royalblue", width=2),
    ))
    fig.update_layout(title=f"{symbol} 策略净值曲线", xaxis_title="日期", yaxis_title="净值")
    st.plotly_chart(fig, use_container_width=True)

# 实验记录
st.subheader("📊 策略迭代记录")
try:
    results_df = pd.read_csv("research/results.tsv", sep="\t")
    st.dataframe(results_df, use_container_width=True)
except FileNotFoundError:
    st.info("尚无实验记录，请先运行策略迭代循环。")
```

```bash
# 启动 Dashboard
uv run streamlit run dashboard/app.py
# 浏览器访问 http://localhost:8501
```

---

## 持续迭代建议

### 短期（1-4 周）

1. **完成 MVP**：回测 + 基础 agent + 迭代循环跑通
2. **跑通 ≥ 50 次自主策略迭代**，观察 Sharpe 趋势
3. **接入纸面交易**，验证信号执行逻辑

### 中期（1-3 个月）

4. **新增 A 股数据源**（Tushare），验证策略在中国市场的适用性
5. **扩充 Agent 类型**：情感分析（新闻 + 财报电话会议）、估值 agent
6. **Walk-Forward 验证**：确保策略不过拟合

### 长期（3 个月+）

7. **LLM 微调**：用历史交易数据微调专用金融语言模型（借助 autoresearch 的 `train.py` 框架）
8. **多策略组合**：多个策略并行运行，通过风险平价分配权重
9. **实盘上线**（需充分纸面交易验证后再考虑）

---

## 常见问题

### Q: autoresearch 的 train.py 和量化策略的关系是什么？

A: 它们的设计哲学完全一致：
- autoresearch：agent 修改 `train.py` → 训练 5 分钟 → 对比 `val_bpb` → 保留/丢弃
- 量化平台：agent 修改 `strategy_template.py` → 回测 → 对比 `Sharpe` → 保留/丢弃

你完全可以用同一套 `program.md` 哲学驱动 AI agent 自主改进策略代码。

### Q: 担心 AI agent 策略过于激进，如何限制风险？

A: 在 `strategy_program.md` 中明确约束：
```markdown
**风控硬约束（agent 不得违反）：**
- 单笔最大亏损 ≤ 总资产 2%
- 策略最大回撤触发 15% 时停止下单
- 不允许使用杠杆
```

### Q: 如何评估策略是否真的有效（非过拟合）？

A: 使用 Walk-Forward 回测（见阶段三步骤 3.3）。确保：
- In-sample Sharpe ≥ 1.0
- Out-of-sample Sharpe ≥ 0.7
- 两者差距 < 0.5

### Q: 想支持加密货币怎么办？

A: 可以用 `ccxt` 库替换 `yfinance`，接口保持 `get_price_history` / `get_fundamentals` 一致即可，其余代码无需改动。

---

## 参考资源

- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — 多 Agent 对冲基金框架原版
- [liuxing95/autoresearch](https://github.com/liuxing95/autoresearch) — 自主迭代研究框架
- [VectorBT 文档](https://vectorbt.dev/) — 高性能向量化回测引擎
- [Alpaca API 文档](https://alpaca.markets/docs/) — 纸面 / 实盘交易 API
- [Tushare Pro 文档](https://tushare.pro/document/2) — A 股专业数据源
- [AKShare 文档](https://akshare.akfamily.xyz/) — 开源 A 股 / 港股数据
