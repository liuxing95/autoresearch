"""
Model ensemble for NASDAQ quantitative trading.
Supports multiple combination methods: equal weight, IC-weighted, rank average.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nasdaq_quant.models.trainer import create_model, prepare_xy


class ModelEnsemble:
    """
    Ensemble of multiple prediction models.

    Supports:
    - equal: simple average of predictions
    - ic_weighted: weight by validation IC (information coefficient)
    - rank_average: average of prediction ranks
    """

    def __init__(
        self,
        model_configs: List[Dict],
        method: str = "ic_weighted",
    ):
        """
        Args:
            model_configs: list of dicts, each with 'type' and model-specific kwargs.
                           e.g. [{"type": "lightgbm", ...}, {"type": "linear"}]
            method: ensemble method - "equal", "ic_weighted", "rank_average"
        """
        self.model_configs = model_configs
        self.method = method
        self.models = []
        self.weights = []
        self.model_names = []

    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        label_col: str = "LABEL0",
        feature_cols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train all models and compute ensemble weights.

        Returns:
            dict with per-model metrics and ensemble info
        """
        all_metrics = {}
        valid_ics = []

        for i, cfg in enumerate(self.model_configs):
            model_type = cfg.pop("type", "lightgbm")
            model_name = f"{model_type}_{i}"
            self.model_names.append(model_name)

            print(f"\n--- Training model [{i+1}/{len(self.model_configs)}]: {model_type} ---")

            # Set d_feat for sequence models
            if model_type in ("lstm", "gru") and feature_cols:
                cfg["d_feat"] = len(feature_cols)

            model = create_model(model_type, **cfg)
            metrics = model.train(train_data, valid_data, label_col=label_col)

            self.models.append(model)
            valid_ic = metrics.get("valid_ic", 0)
            valid_ics.append(valid_ic)

            all_metrics[model_name] = metrics

            # Restore type for potential re-use
            cfg["type"] = model_type

        # Compute weights
        if self.method == "equal":
            n = len(self.models)
            self.weights = [1.0 / n] * n
        elif self.method == "ic_weighted":
            # Use max(ic, 0) to avoid negative weights
            positive_ics = [max(ic, 0) for ic in valid_ics]
            total = sum(positive_ics)
            if total > 0:
                self.weights = [ic / total for ic in positive_ics]
            else:
                n = len(self.models)
                self.weights = [1.0 / n] * n
        elif self.method == "rank_average":
            # Equal weights for rank averaging (handled in predict)
            n = len(self.models)
            self.weights = [1.0 / n] * n
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        print(f"\n--- Ensemble Weights ({self.method}) ---")
        for name, w in zip(self.model_names, self.weights):
            print(f"  {name}: {w:.4f}")

        all_metrics["ensemble_method"] = self.method
        all_metrics["ensemble_weights"] = dict(zip(self.model_names, self.weights))

        return all_metrics

    def predict(
        self,
        dataset: pd.DataFrame,
        label_col: str = "LABEL0",
    ) -> pd.Series:
        """Generate ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(dataset, label_col=label_col)
            predictions.append(pred)

        if self.method == "rank_average":
            # Average the ranks instead of raw scores
            rank_preds = []
            for pred in predictions:
                # Rank per date (cross-sectional ranking)
                ranked = pred.groupby(level="date").rank(pct=True)
                rank_preds.append(ranked)

            ensemble_pred = sum(w * rp for w, rp in zip(self.weights, rank_preds))
        else:
            # Weighted average of raw scores
            ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))

        ensemble_pred.name = "pred_score"
        return ensemble_pred

    def save(self, save_dir: str) -> None:
        """Save all models."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in zip(self.model_names, self.models):
            model_type = name.rsplit("_", 1)[0]
            ext = ".pt" if model_type in ("lstm", "gru") else ".pkl"
            path = os.path.join(save_dir, f"ensemble_{name}{ext}")
            model.save(path)

        # Save ensemble config
        import json
        config = {
            "method": self.method,
            "weights": dict(zip(self.model_names, self.weights)),
            "model_names": self.model_names,
        }
        config_path = os.path.join(save_dir, "ensemble_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble config saved to {config_path}")
