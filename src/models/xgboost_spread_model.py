"""
XGBoost Spread Change Model
CS3244 Machine Learning - Group 23

Predicts 10-day spread CHANGE (not raw value) for both OLS and Kalman spreads.

Two variants:
    xgboost_ols    — target: label_continuous_10d = spread_ols(t+10)    - spread_ols(t)
    xgboost_kalman — target: label_kalman_10d     = spread_kalman(t+10) - spread_kalman(t)

Both variants use the same 11 input features (see FEATURE_NAMES).
The OLS/Kalman distinction only affects the prediction TARGET, not the inputs.

Outputs per prediction row:
    predicted_change  — what the model directly predicts
    predicted_value   — current_spread + predicted_change
    predicted_z       — predicted_change / rolling_vol_20d

Hyperparameter tuning: grid search over 18 combos per MODEL_BRIEF
    max_depth    : [3, 4, 5]
    n_estimators : [100, 200]
    learning_rate: [0.01, 0.05, 0.1]
    Metric: average MSE across validation windows (minimise)

Evaluation metrics:
    MSE                  — primary
    MAE                  — secondary
    Directional accuracy — sanity check (should be > 50%)
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── 11 input features — must match get_feature_columns() in feature_engineering.py ──
FEATURE_NAMES: List[str] = [
    "z_score",              # OLS spread z-score (60d rolling mean/std)
    "z_score_kalman",       # Kalman spread z-score
    "momentum_5d",          # spread change over last 5 days
    "momentum_10d",         # spread change over last 10 days
    "rolling_vol_20d",      # 20-day rolling volatility of spread
    "rolling_vol_60d",      # 60-day rolling volatility of spread
    "rolling_corr_60d",     # 60-day rolling correlation of the two legs
    "days_since_crossing",  # days since spread last crossed its rolling mean
    "kalman_beta",          # current Kalman hedge ratio
    "kalman_beta_change",   # 5-day change in Kalman beta
    "spread_acceleration",  # second derivative of spread
]

# Target column names — produced by feature_engineering.py / pair_dataset_builder.py
TARGET_OLS    = "label_continuous_10d"   # spread_ols(t+10)    - spread_ols(t)
TARGET_KALMAN = "label_kalman_10d"       # spread_kalman(t+10) - spread_kalman(t)

# Spread columns for predicted_value derivation
SPREAD_COL_OLS    = "spread_ols"
SPREAD_COL_KALMAN = "spread_kalman"

# Grid per MODEL_BRIEF — 18 combos total
PARAM_GRID = {
    "max_depth":     [3, 4, 5],
    "n_estimators":  [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core model
# ─────────────────────────────────────────────────────────────────────────────

class SpreadChangeXGBoost:
    """
    XGBoost regressor that predicts 10-day spread change.

    INPUT  : feature matrix (n_samples, 11)
    OUTPUT : predicted spread change (continuous float) via predict()

    After predict(), use derive_outputs() to get the full
    (predicted_change, predicted_value, predicted_z) triple.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        **kwargs,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=42,
            **kwargs,
        )
        self.is_trained = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "SpreadChangeXGBoost":
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.is_trained = True
        if verbose:
            print(f"  ✓ Trained on {len(X_train):,} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted spread change (continuous floats)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)

    @staticmethod
    def derive_outputs(
        predicted_change: np.ndarray,
        current_spread: np.ndarray,
        rolling_vol_20d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive all three output columns from the raw model prediction.

            predicted_value = current_spread + predicted_change
            predicted_z     = predicted_change / rolling_vol_20d

        Args:
            predicted_change : raw model output (n,)
            current_spread   : spread_ols[t] or spread_kalman[t]  (n,)
            rolling_vol_20d  : rolling_vol_20d feature column (n,)

        Returns:
            (predicted_change, predicted_value, predicted_z)
        """
        predicted_value = current_spread + predicted_change
        predicted_z     = predicted_change / (rolling_vol_20d + 1e-8)
        return predicted_change, predicted_value, predicted_z


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    actual_change: np.ndarray,
    predicted_change: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all three evaluation metrics from MODEL_BRIEF.

    Args:
        actual_change    : label_continuous_10d or label_kalman_10d  (n,)
        predicted_change : model output  (n,)

    Returns:
        {"mse": float, "mae": float, "directional_accuracy": float}
    """
    mse = float(mean_squared_error(actual_change, predicted_change))
    mae = float(mean_absolute_error(actual_change, predicted_change))

    nonzero = actual_change != 0
    if nonzero.sum() > 0:
        dir_acc = float(
            np.mean(np.sign(predicted_change[nonzero]) == np.sign(actual_change[nonzero]))
        )
    else:
        dir_acc = float("nan")

    return {"mse": mse, "mae": mae, "directional_accuracy": dir_acc}


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostPipeline:
    """
    Full training pipeline for one spread variant (OLS or Kalman).

    Responsibilities:
    - Stack features from all pairs into one global model (per MODEL_BRIEF)
    - Grid search hyperparameter tuning (18 combos)
    - Train final model with best params
    - Evaluate: MSE / MAE / directional accuracy
    - SHAP feature importance
    """

    def __init__(self, spread_type: str = "ols"):
        """
        Args:
            spread_type: 'ols' or 'kalman'
        """
        if spread_type not in ("ols", "kalman"):
            raise ValueError("spread_type must be 'ols' or 'kalman'")
        self.spread_type = spread_type
        self.target_col  = TARGET_OLS if spread_type == "ols" else TARGET_KALMAN
        self.spread_col  = SPREAD_COL_OLS if spread_type == "ols" else SPREAD_COL_KALMAN
        self.model_name  = f"xgboost_{spread_type}"
        self.best_model: Optional[SpreadChangeXGBoost] = None
        self.best_params: Optional[Dict] = None

    # ── data stacking ─────────────────────────────────────────────────────

    def stack_pairs(
        self,
        X_dict: Dict[str, np.ndarray],
        y_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenate per-pair arrays into single matrices.

        Args:
            X_dict: {pair → (n, 11) array}
            y_dict: {pair → (n,) array of spread change targets}

        Returns:
            (X, y) as concatenated numpy arrays
        """
        X = np.concatenate(list(X_dict.values()), axis=0)
        y = np.concatenate(list(y_dict.values()), axis=0)
        return X, y

    # ── grid search ───────────────────────────────────────────────────────

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """
        Grid search over 18 combos per MODEL_BRIEF.

        Grid:
            max_depth    : [3, 4, 5]
            n_estimators : [100, 200]
            learning_rate: [0.01, 0.05, 0.1]

        Selects by lowest validation MSE.

        Returns:
            {"results_df": DataFrame, "best_params": dict}
        """
        combos = list(itertools.product(
            PARAM_GRID["max_depth"],
            PARAM_GRID["n_estimators"],
            PARAM_GRID["learning_rate"],
        ))

        print(f"\n  Grid search ({len(combos)} combos) — {self.model_name}")
        print(f"  {'depth':>5}  {'n_est':>5}  {'lr':>5}  {'val_mse':>10}  {'dir_acc':>8}")
        print(f"  {'-'*45}")

        rows = []
        for depth, n_est, lr in combos:
            m = SpreadChangeXGBoost(max_depth=depth, n_estimators=n_est, learning_rate=lr)
            m.fit(X_train, y_train, X_val, y_val, verbose=False)
            preds   = m.predict(X_val)
            metrics = evaluate_predictions(y_val, preds)
            rows.append({
                "max_depth":     depth,
                "n_estimators":  n_est,
                "learning_rate": lr,
                "val_mse":       metrics["mse"],
                "val_mae":       metrics["mae"],
                "val_dir_acc":   metrics["directional_accuracy"],
            })
            print(
                f"  {depth:>5}  {n_est:>5}  {lr:>5.2f}  "
                f"{metrics['mse']:>10.6f}  {metrics['directional_accuracy']:>8.3f}"
            )

        results_df = pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)
        best_row   = results_df.iloc[0]
        best_params = {
            "max_depth":     int(best_row["max_depth"]),
            "n_estimators":  int(best_row["n_estimators"]),
            "learning_rate": float(best_row["learning_rate"]),
            "val_mse":       float(best_row["val_mse"]),
            "val_mae":       float(best_row["val_mae"]),
            "val_dir_acc":   float(best_row["val_dir_acc"]),
        }

        print(
            f"\n  Best: depth={best_params['max_depth']}  "
            f"n_est={best_params['n_estimators']}  "
            f"lr={best_params['learning_rate']}  "
            f"val_mse={best_params['val_mse']:.6f}"
        )
        return {"results_df": results_df, "best_params": best_params}

    # ── train ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **hyperparams,
    ) -> SpreadChangeXGBoost:
        """Train with given hyperparameters; store as self.best_model."""
        defaults = dict(n_estimators=200, max_depth=4, learning_rate=0.05)
        defaults.update(hyperparams)
        self.best_model = SpreadChangeXGBoost(**defaults)
        self.best_model.fit(X_train, y_train, X_val, y_val, verbose=True)
        return self.best_model

    # ── evaluate ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y_actual: np.ndarray,
        model: Optional[SpreadChangeXGBoost] = None,
    ) -> Dict[str, float]:
        """Evaluate a model. Uses self.best_model if model not provided."""
        if model is None:
            model = self.best_model
        preds = model.predict(X)
        return evaluate_predictions(y_actual, preds)

    # ── SHAP ──────────────────────────────────────────────────────────────

    def shap_analysis(
        self,
        X_val: np.ndarray,
        num_samples: int = 500,
        save_dir: str = ".",
    ) -> pd.DataFrame:
        """SHAP feature importance for the trained model."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        explainer   = shap.TreeExplainer(self.best_model.model)
        X_sample    = X_val[:min(num_samples, len(X_val))]
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample, feature_names=FEATURE_NAMES,
            show=False, plot_type="bar",
        )
        plt.title(f"Feature Importance — {self.model_name}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/shap_{self.model_name}_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES, show=False)
        plt.title(f"SHAP Summary — {self.model_name}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/shap_{self.model_name}_detail.png", dpi=300, bbox_inches="tight")
        plt.close()

        importance_df = (
            pd.DataFrame({
                "feature":   FEATURE_NAMES,
                "mean_shap": np.abs(shap_values).mean(axis=0),
            })
            .sort_values("mean_shap", ascending=False)
            .reset_index(drop=True)
        )
        print(f"\n  Feature importance ({self.model_name}):")
        print(importance_df.to_string(index=False))
        return importance_df
