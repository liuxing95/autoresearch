"""
Global configuration for NASDAQ Quantitative Trading System.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB_DATA_DIR = os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "nasdaq_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "quant_results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")

for d in [RESULTS_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# NASDAQ 100 components (top ~100 stocks)
# ---------------------------------------------------------------------------
NASDAQ100_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "NFLX", "ADBE", "AMD", "PEP", "CSCO", "TMUS", "INTC", "CMCSA", "INTU", "TXN",
    "QCOM", "AMGN", "AMAT", "HON", "ISRG", "BKNG", "LRCX", "VRTX", "ADP", "REGN",
    "SBUX", "MDLZ", "ADI", "GILD", "MU", "PANW", "SNPS", "KLAC", "CDNS", "PYPL",
    "MELI", "ABNB", "CRWD", "CSX", "MAR", "CEG", "ORLY", "CTAS", "NXPI", "PCAR",
    "MRVL", "WDAY", "ROST", "ROP", "FTNT", "CPRT", "MNST", "FAST", "DXCM",
    "ODFL", "PAYX", "KDP", "GEHC", "IDXX", "KHC", "FANG", "EXC", "LULU", "CCEP",
    "VRSK", "CTSH", "AEP", "ON", "MCHP", "BKR", "XEL", "ANSS", "DLTR",
    "TTWO", "CDW", "CSGP", "BIIB", "ZS", "WBD",
    "TEAM", "ALGN", "ENPH", "WBA", "JD", "PDD", "MRNA", "BIDU",
]

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: NASDAQ100_SYMBOLS)
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    data_dir: str = QLIB_DATA_DIR
    freq: str = "day"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class LGBConfig:
    loss: str = "mse"
    colsample_bytree: float = 0.8879
    learning_rate: float = 0.0421
    subsample: float = 0.8789
    lambda_l1: float = 1.0
    lambda_l2: float = 1.0
    max_depth: int = 6
    num_leaves: int = 64
    num_threads: int = 10
    n_estimators: int = 1000
    early_stopping_rounds: int = 50


@dataclass
class LSTMConfig:
    d_feat: int = 158
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    n_epochs: int = 100
    lr: float = 0.001
    early_stop: int = 20
    batch_size: int = 2000
    loss: str = "mse"


@dataclass
class ModelConfig:
    model_type: str = "lightgbm"  # "lightgbm", "lstm", "gru", "linear"
    lgb: LGBConfig = field(default_factory=LGBConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    label_col: str = "LABEL0"
    # Time splits
    train_start: str = "2015-01-01"
    train_end: str = "2021-12-31"
    valid_start: str = "2022-01-01"
    valid_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    topk: int = 30
    n_drop: int = 5
    buy_cost: float = 0.001   # 0.1%
    sell_cost: float = 0.001  # 0.1%
    slippage: float = 0.001   # 0.1% slippage
    init_cash: float = 1_000_000.0
    benchmark: str = "^NDX"
    rebalance_freq: str = "day"  # "day" or "week"
    max_weight: float = 0.10  # max 10% per stock
    stop_loss: float = -0.08  # per-stock stop loss threshold (-8%)
    max_drawdown_limit: float = -0.20  # portfolio drawdown circuit breaker (-20%)


# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    enable: bool = False
    models: List[str] = field(default_factory=lambda: ["lightgbm", "linear"])
    method: str = "ic_weighted"  # "equal", "ic_weighted", "rank_average"


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    results_dir: str = RESULTS_DIR
    model_dir: str = MODEL_DIR
    report_dir: str = REPORT_DIR
