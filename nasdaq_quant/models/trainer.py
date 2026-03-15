"""
Model trainer for NASDAQ quantitative trading.
Supports LightGBM, Linear, and PyTorch-based models (LSTM, GRU).
"""

import os
import time
import pickle
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Utility: extract X/y from multi-index DataFrame
# ---------------------------------------------------------------------------

def prepare_xy(
    dataset: pd.DataFrame,
    label_col: str = "LABEL0",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Separate features and label from dataset.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label array (n_samples,)
        info: DataFrame with date and symbol index
    """
    feature_cols = [c for c in dataset.columns if c != label_col]
    X = dataset[feature_cols].values.astype(np.float32)
    y = dataset[label_col].values.astype(np.float32)
    info = dataset[[label_col]].copy()
    return X, y, info


# ---------------------------------------------------------------------------
# LightGBM Model
# ---------------------------------------------------------------------------

class LightGBMModel:
    """LightGBM gradient boosting tree model."""

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.0421,
        max_depth: int = 8,
        num_leaves: int = 210,
        subsample: float = 0.8789,
        colsample_bytree: float = 0.8879,
        lambda_l1: float = 205.6999,
        lambda_l2: float = 580.9768,
        early_stopping_rounds: int = 50,
        num_threads: int = 10,
        verbose: int = -1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": lambda_l1,
            "reg_lambda": lambda_l2,
            "n_jobs": num_threads,
            "verbose": verbose,
            "random_state": 42,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        label_col: str = "LABEL0",
    ) -> Dict:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install: pip install lightgbm")

        X_train, y_train, _ = prepare_xy(train_data, label_col)
        X_valid, y_valid, _ = prepare_xy(valid_data, label_col)

        print(f"Training LightGBM: train={X_train.shape}, valid={X_valid.shape}")
        t0 = time.time()

        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=callbacks,
        )

        dt = time.time() - t0
        train_pred = self.model.predict(X_train)
        valid_pred = self.model.predict(X_valid)

        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "valid_mse": mean_squared_error(y_valid, valid_pred),
            "train_ic": np.corrcoef(y_train, train_pred)[0, 1],
            "valid_ic": np.corrcoef(y_valid, valid_pred)[0, 1],
            "best_iteration": self.model.best_iteration_,
            "training_time": dt,
        }

        print(f"\nLightGBM training complete in {dt:.1f}s")
        print(f"  Train MSE: {metrics['train_mse']:.6f}, IC: {metrics['train_ic']:.4f}")
        print(f"  Valid MSE: {metrics['valid_mse']:.6f}, IC: {metrics['valid_ic']:.4f}")
        print(f"  Best iteration: {metrics['best_iteration']}")

        return metrics

    def predict(self, dataset: pd.DataFrame, label_col: str = "LABEL0") -> pd.Series:
        X, _, info = prepare_xy(dataset, label_col)
        preds = self.model.predict(X)
        return pd.Series(preds, index=dataset.index, name="pred_score")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

    def feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Linear Model
# ---------------------------------------------------------------------------

class LinearModel:
    """Simple Ridge regression baseline."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        label_col: str = "LABEL0",
    ) -> Dict:
        X_train, y_train, _ = prepare_xy(train_data, label_col)
        X_valid, y_valid, _ = prepare_xy(valid_data, label_col)

        print(f"Training Ridge: train={X_train.shape}, valid={X_valid.shape}")
        t0 = time.time()

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)

        dt = time.time() - t0
        train_pred = self.model.predict(X_train)
        valid_pred = self.model.predict(X_valid)

        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "valid_mse": mean_squared_error(y_valid, valid_pred),
            "train_ic": np.corrcoef(y_train, train_pred)[0, 1],
            "valid_ic": np.corrcoef(y_valid, valid_pred)[0, 1],
            "training_time": dt,
        }

        print(f"\nRidge training complete in {dt:.1f}s")
        print(f"  Train MSE: {metrics['train_mse']:.6f}, IC: {metrics['train_ic']:.4f}")
        print(f"  Valid MSE: {metrics['valid_mse']:.6f}, IC: {metrics['valid_ic']:.4f}")

        return metrics

    def predict(self, dataset: pd.DataFrame, label_col: str = "LABEL0") -> pd.Series:
        X, _, _ = prepare_xy(dataset, label_col)
        preds = self.model.predict(X)
        return pd.Series(preds, index=dataset.index, name="pred_score")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)


# ---------------------------------------------------------------------------
# LSTM/GRU Model (PyTorch)
# ---------------------------------------------------------------------------

class SequenceModel:
    """LSTM/GRU model for sequential stock prediction."""

    def __init__(
        self,
        model_type: str = "lstm",  # "lstm" or "gru"
        d_feat: int = 158,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 100,
        lr: float = 0.001,
        early_stop: int = 20,
        batch_size: int = 2000,
        seq_len: int = 20,
        device: str = "auto",
    ):
        self.model_type = model_type
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model = None

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

    def _build_model(self):
        import torch
        import torch.nn as nn

        class _SeqModel(nn.Module):
            def __init__(self, model_type, d_feat, hidden, num_layers, dropout):
                super().__init__()
                RNN = nn.LSTM if model_type == "lstm" else nn.GRU
                self.rnn = RNN(
                    input_size=d_feat,
                    hidden_size=hidden,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: (batch, seq_len, d_feat)
                out, _ = self.rnn(x)
                # Use last timestep
                out = out[:, -1, :]
                return self.fc(out).squeeze(-1)

        return _SeqModel(
            self.model_type, self.d_feat, self.hidden_size,
            self.num_layers, self.dropout,
        )

    def _prepare_sequences(
        self, dataset: pd.DataFrame, label_col: str = "LABEL0"
    ) -> Tuple:
        """
        Prepare sequential data: for each (date, symbol), look back seq_len days.
        """
        import torch

        feature_cols = [c for c in dataset.columns if c != label_col]
        self.d_feat = len(feature_cols)

        dates = dataset.index.get_level_values("date").unique().sort_values()
        symbols = dataset.index.get_level_values("symbol").unique()

        X_list = []
        y_list = []
        idx_list = []

        for symbol in symbols:
            sym_data = dataset.xs(symbol, level="symbol") if symbol in dataset.index.get_level_values("symbol") else None
            if sym_data is None or len(sym_data) < self.seq_len:
                continue

            feat_vals = sym_data[feature_cols].values
            label_vals = sym_data[label_col].values
            sym_dates = sym_data.index

            for i in range(self.seq_len, len(sym_data)):
                seq = feat_vals[i - self.seq_len:i]
                X_list.append(seq)
                y_list.append(label_vals[i])
                idx_list.append((sym_dates[i], symbol))

        if not X_list:
            return None, None, None

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        idx = pd.MultiIndex.from_tuples(idx_list, names=["date", "symbol"])

        return X, y, idx

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        label_col: str = "LABEL0",
    ) -> Dict:
        import torch
        import torch.nn as nn

        X_train, y_train, _ = self._prepare_sequences(train_data, label_col)
        X_valid, y_valid, _ = self._prepare_sequences(valid_data, label_col)

        if X_train is None:
            raise ValueError("Not enough sequential data for training.")

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        print(f"Training {self.model_type.upper()}: train={X_train.shape}, valid={X_valid.shape}, device={self.device}")
        t0 = time.time()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0
            n_batches = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_valid.to(self.device))
                val_loss = criterion(val_pred, y_valid.to(self.device)).item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        dt = time.time() - t0

        # Compute metrics
        self.model.eval()
        with torch.no_grad():
            t_pred = self.model(X_train.to(self.device)).cpu().numpy()
            v_pred = self.model(X_valid.to(self.device)).cpu().numpy()

        y_t = y_train.numpy()
        y_v = y_valid.numpy()

        metrics = {
            "train_mse": mean_squared_error(y_t, t_pred),
            "valid_mse": mean_squared_error(y_v, v_pred),
            "train_ic": np.corrcoef(y_t, t_pred)[0, 1] if len(y_t) > 1 else 0,
            "valid_ic": np.corrcoef(y_v, v_pred)[0, 1] if len(y_v) > 1 else 0,
            "training_time": dt,
        }

        print(f"\n{self.model_type.upper()} training complete in {dt:.1f}s")
        print(f"  Train MSE: {metrics['train_mse']:.6f}, IC: {metrics['train_ic']:.4f}")
        print(f"  Valid MSE: {metrics['valid_mse']:.6f}, IC: {metrics['valid_ic']:.4f}")

        return metrics

    def predict(self, dataset: pd.DataFrame, label_col: str = "LABEL0") -> pd.Series:
        import torch

        X, _, idx = self._prepare_sequences(dataset, label_col)
        if X is None:
            return pd.Series(dtype=float, name="pred_score")

        self.model.eval()
        preds_list = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size].to(self.device)
                pred = self.model(xb).cpu().numpy()
                preds_list.append(pred)

        preds = np.concatenate(preds_list)
        return pd.Series(preds, index=idx, name="pred_score")

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "model_type": self.model_type,
                "d_feat": self.d_feat,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
        }, path)

    def load(self, path: str) -> None:
        import torch
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]
        self.model_type = config["model_type"]
        self.d_feat = config["d_feat"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(model_type: str, **kwargs):
    """Factory function to create model instance."""
    if model_type == "lightgbm":
        return LightGBMModel(**kwargs)
    elif model_type == "linear":
        return LinearModel(**kwargs)
    elif model_type in ("lstm", "gru"):
        return SequenceModel(model_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
