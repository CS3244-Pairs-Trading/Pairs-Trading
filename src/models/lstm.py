"""
LSTM Spread Prediction Model
==============================
Predicts 10-day spread change using sequences of engineered features.
Supports both OLS and Kalman spread targets.

Architecture
------------
2-layer LSTM → Linear(hidden, 1) → predicted_spread_change

The model sees a sliding window of the last N days of 11 features for
one pair and predicts how much the spread will change over the next 10
days. Training is GLOBAL -- one model trained on all pairs pooled together.
Hyperparameters (hidden_size, window_size, learning_rate) are tuned by
averaging validation MSE across the 4 sliding folds.

Hyperparameter grid
-------------------
    hidden_size:    [32, 64]
    window_size:    [10, 20]
    learning_rate:  [0.001, 0.01]
    num_layers:     2          (fixed -- deeper overfits on ~15k rows)
    dropout:        0.2        (fixed)
    batch_size:     128        (fixed)
    epochs:         50         (fixed, with early stopping patience=7)

= 8 combinations x 4 folds = 32 training runs per spread type.

Input
-----
    data/processed/pair_datasets/<window>/train_pair_dataset.csv
    data/processed/pair_datasets/<window>/val_pair_dataset.csv

Output
------
    data/processed/predictions/lstm_ols/<window>/predictions.csv
    data/processed/predictions/lstm_kalman/<window>/predictions.csv

    Columns: Date, pair, predicted_change, predicted_value, predicted_z

Model saving (NEW)
------------------
    After each window's train_model() call the best weights are persisted via
    torch.save() so SHAP / GradientExplainer analysis can reload the model
    without any retraining.

    Saved to:
        data/processed/models/lstm_<spread_type>/<window>.pt

    Companion metadata (feature mu/std needed for inference):
        data/processed/models/lstm_<spread_type>/<window>_norm.npz

Usage
-----
    # Full run (tune + predict on all folds for both OLS and Kalman)
    python3 -m src.models.lstm

    # Single window, OLS only
    python3 -m src.models.lstm --window 2010_2012 --spread ols

    # Skip tuning, use specific hyperparameters
    python3 -m src.models.lstm --hidden 64 --window_size 20 --lr 0.001 --no_tune
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_CONFIG
from src.models.prediction_metrics import evaluate_regression_predictions

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "z_score",
    "z_score_kalman",
    "momentum_5d",
    "momentum_10d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "rolling_corr_60d",
    "days_since_crossing",
    "kalman_beta",
    "kalman_beta_change",
    "spread_acceleration",
]

# Fixed training params (not tuned)
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 7  # early stopping

# Hyperparameter grid
HP_GRID = {
    "hidden_size": [32, 64],
    "window_size": [10, 20],
    "learning_rate": [0.001, 0.01],
}


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class SpreadLSTM(nn.Module):
    """
    2-layer LSTM for spread change prediction.

    Input:  (batch, window_size, n_features)
    Output: (batch,) -- predicted spread change
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # take last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_hidden).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# NEW: save / load helpers
# ---------------------------------------------------------------------------

def save_lstm_model(
    model: SpreadLSTM,
    feat_mu: np.ndarray,
    feat_std: np.ndarray,
    path: str | Path,
) -> None:
    """
    Persist a trained SpreadLSTM and its normalisation statistics.

    Two files are written next to each other:
        <path>.pt   — torch state dict (weights only, architecture-agnostic)
        <path>_norm.npz — feat_mu and feat_std arrays needed for inference

    Args:
        model    : trained SpreadLSTM (CPU or GPU)
        feat_mu  : per-feature training mean  (shape: n_features,)
        feat_std : per-feature training std   (shape: n_features,)
        path     : destination path WITHOUT extension,
                   e.g. "data/processed/models/lstm_ols/2011_2013"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always save weights on CPU so the file is device-independent
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, str(path) + ".pt")
    np.savez(str(path) + "_norm.npz", feat_mu=feat_mu, feat_std=feat_std)
    print(f"  ✓ LSTM saved → {path}.pt + {path}_norm.npz")


def load_lstm_model(
    path: str | Path,
    hidden_size: int,
    input_size: int = 11,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
) -> tuple[SpreadLSTM, np.ndarray, np.ndarray]:
    """
    Restore a saved SpreadLSTM and its normalisation statistics.

    Args:
        path        : same path stem passed to save_lstm_model()
        hidden_size : must match the value used when training
        input_size  : number of input features (default 11)
        num_layers  : must match the value used when training
        dropout     : must match the value used when training

    Returns:
        (model, feat_mu, feat_std)
        where model is in eval() mode on CPU.
    """
    path = Path(path)
    model = SpreadLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(str(path) + ".pt", map_location="cpu"))
    model.eval()

    norm = np.load(str(path) + "_norm.npz")
    print(f"  ✓ LSTM loaded ← {path}.pt + {path}_norm.npz")
    return model, norm["feat_mu"], norm["feat_std"]


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def load_pair_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a pair dataset CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["pair", "Date"]).reset_index(drop=True)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Build sliding-window sequences from pair dataset.

    Groups by pair, sorts by date, creates overlapping windows.
    Returns X (N, window_size, n_features), y (N,),
    dates (N,) and pairs (N,) for the prediction rows.
    """
    X_all, y_all, dates_all, pairs_all = [], [], [], []

    for pair_name, pair_df in df.groupby("pair", sort=True):
        pair_df = pair_df.sort_values("Date").reset_index(drop=True)

        # Check all feature columns exist
        missing = [c for c in feature_cols if c not in pair_df.columns]
        if missing:
            warnings.warn(f"Pair {pair_name}: missing features {missing}, skipping.")
            continue

        if target_col not in pair_df.columns:
            warnings.warn(f"Pair {pair_name}: missing target '{target_col}', skipping.")
            continue

        X = pair_df[feature_cols].values  # (T, n_features)
        y = pair_df[target_col].values  # (T,)
        dates = pair_df["Date"].dt.strftime("%Y-%m-%d").values
        n = len(X)

        if n <= window_size:
            continue

        for i in range(window_size, n):
            x_seq = X[i - window_size : i]  # (window_size, n_features)
            y_val = y[i]

            # Skip if any NaN in features or target
            if np.isnan(x_seq).any() or np.isnan(y_val):
                continue

            X_all.append(x_seq)
            y_all.append(y_val)
            dates_all.append(dates[i])
            pairs_all.append(pair_name)

    if not X_all:
        return np.array([]), np.array([]), [], []

    return (
        np.array(X_all, dtype=np.float32),
        np.array(y_all, dtype=np.float32),
        dates_all,
        pairs_all,
    )


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-normalize features using training set statistics.

    Normalizes per feature across all timesteps and samples.
    Returns normalized arrays + mean/std for inverse transform.
    """
    # Reshape to (N * window, features) for per-feature stats
    n_train, w, f = X_train.shape
    flat = X_train.reshape(-1, f)
    mu = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero

    X_train_norm = (X_train - mu) / std
    X_val_norm = (X_val - mu) / std

    return X_train_norm, X_val_norm, mu, std


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    verbose: bool = False,
) -> tuple[SpreadLSTM, float]:
    """
    Train LSTM with early stopping on validation MSE.

    Returns (trained_model, best_val_mse).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[2]

    model = SpreadLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_X_t = torch.from_numpy(X_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_mse = loss_fn(val_pred, val_y_t).item()

        if verbose and (epoch + 1) % 10 == 0:
            train_mse = epoch_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1:3d}: train_mse={train_mse:.6f}, val_mse={val_mse:.6f}")

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_val_mse


def predict(
    model: SpreadLSTM,
    X: np.ndarray,
) -> np.ndarray:
    """Run inference, return predicted_change array."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        preds = model(X_t).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, directional accuracy, R², IC, profit-weighted DA, DW-MSE."""
    metrics = evaluate_regression_predictions(actual, predicted)

    return {
        "rmse": metrics["rmse"],
        "dir_acc": metrics["directional_accuracy"],
        "r2": metrics["r2"],
        "information_coefficient": metrics["information_coefficient"],
        "profit_weighted_da": metrics["profit_weighted_da"],
        "directional_weighted_mse": metrics["directional_weighted_mse"],
    }


# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    pair_dataset_root: Path,
    target_col: str,
    verbose: bool = True,
) -> dict:
    """
    Grid search over HP_GRID, averaging validation MSE across all available
    sliding folds. Returns the best hyperparameter combination.
    """
    config = DEFAULT_CONFIG
    folds = config.expanding_folds  # 4 sliding validation folds

    grid = list(product(
        HP_GRID["hidden_size"],
        HP_GRID["window_size"],
        HP_GRID["learning_rate"],
    ))

    results = []

    for hidden, window, lr in grid:
        fold_metrics = []
        if verbose:
            print(f"\n  Tuning: hidden={hidden}, window={window}, lr={lr}")

        for fold in folds:
            train_path = pair_dataset_root / fold.label / "train_pair_dataset.csv"
            val_path = pair_dataset_root / fold.label / "val_pair_dataset.csv"

            if not train_path.exists() or not val_path.exists():
                if verbose:
                    print(f"    Fold {fold.label}: SKIP (files missing)")
                continue

            train_df = load_pair_dataset(train_path)
            val_df = load_pair_dataset(val_path)

            X_train, y_train, _, _ = build_sequences(
                train_df, FEATURE_COLS, target_col, window
            )
            X_val, y_val, _, _ = build_sequences(
                val_df, FEATURE_COLS, target_col, window
            )

            if len(X_train) == 0 or len(X_val) == 0:
                if verbose:
                    print(f"    Fold {fold.label}: SKIP (no sequences)")
                continue

            # Normalize
            X_train_n, X_val_n, _, _ = normalize_features(X_train, X_val)

            # Train
            model, val_mse = train_model(
                X_train_n, y_train, X_val_n, y_val,
                hidden_size=hidden,
                learning_rate=lr,
                verbose=False,
            )

            # Full metrics for composite scoring
            preds = predict(model, X_val_n)
            metrics = compute_metrics(y_val, preds)
            fold_metrics.append(metrics)
            if verbose:
                print(f"    Fold {fold.label}: RMSE={metrics['rmse']:.6f}  "
                      f"R²={metrics['r2']:.4f}  DirAcc={metrics['dir_acc']:.3f}  "
                      f"(train={len(X_train)}, val={len(X_val)} sequences)")

        if not fold_metrics:
            avg_rmse = float("inf")
            avg_dw_mse = float("inf")
            avg_r2 = float("-inf")
            avg_dir_acc = 0.0
            std_rmse = float("nan")
        else:
            avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
            avg_dw_mse = np.mean([m["directional_weighted_mse"] for m in fold_metrics])
            avg_r2 = np.mean([m["r2"] for m in fold_metrics])
            avg_dir_acc = np.mean([m["dir_acc"] for m in fold_metrics])
            std_rmse = np.std([m["rmse"] for m in fold_metrics])

        composite = 0.5 * avg_r2 + 0.5 * avg_dir_acc

        results.append({
            "hidden_size": hidden,
            "window_size": window,
            "learning_rate": lr,
            "avg_val_rmse": avg_rmse,
            "avg_val_directional_weighted_mse": avg_dw_mse,
            "avg_val_r2": avg_r2,
            "avg_val_dir_acc": avg_dir_acc,
            "composite_score": composite,
            "std_val_rmse": std_rmse,
            "n_folds": len(fold_metrics),
        })

        if verbose:
            print(f"    Avg RMSE={avg_rmse:.6f}  R²={avg_r2:.4f}  "
                  f"DirAcc={avg_dir_acc:.3f}  Composite={composite:.4f}  "
                  f"({len(fold_metrics)} folds)")

    # Pick best by composite score (R² + directional accuracy)
    results_df = pd.DataFrame(results).sort_values(
        ["composite_score", "avg_val_directional_weighted_mse", "avg_val_rmse"],
        ascending=[False, True, True],
    )
    best = results_df.iloc[0]

    if verbose:
        print(f"\n{'='*60}")
        print("TUNING RESULTS:")
        print(results_df.to_string(index=False))
        print(f"\nBest: hidden={int(best['hidden_size'])}, "
              f"window={int(best['window_size'])}, "
              f"lr={best['learning_rate']}, "
              f"DW-MSE={best['avg_val_directional_weighted_mse']:.6f}, "
              f"R²={best['avg_val_r2']:.4f}, "
              f"DirAcc={best['avg_val_dir_acc']:.3f}, "
              f"Composite={best['composite_score']:.4f}")
        print(f"{'='*60}")

    return {
        "hidden_size": int(best["hidden_size"]),
        "window_size": int(best["window_size"]),
        "learning_rate": float(best["learning_rate"]),
        "avg_val_rmse": float(best["avg_val_rmse"]),
        "avg_val_directional_weighted_mse": float(best["avg_val_directional_weighted_mse"]),
        "tuning_results": results_df,
    }


# ---------------------------------------------------------------------------
# PREDICTION + OUTPUT
# ---------------------------------------------------------------------------

def run_predictions_for_window(
    pair_dataset_root: Path,
    window_label: str,
    target_col: str,
    spread_col: str,
    hidden_size: int,
    window_size: int,
    learning_rate: float,
    output_root: Path,
    is_holdout: bool = False,
    verbose: bool = True,
    models_root: Path | None = None,   # NEW
) -> dict[str, float] | None:
    """
    Train on one window's train set, predict on val/test set,
    save unified predictions CSV, return metrics.

    NEW: saves the trained model weights and normalisation stats to
         models_root / lstm_<spread_type> / <window_label>.pt
         models_root / lstm_<spread_type> / <window_label>_norm.npz
    """
    train_path = pair_dataset_root / window_label / "train_pair_dataset.csv"
    eval_name = "test" if is_holdout else "val"
    eval_path = pair_dataset_root / window_label / f"{eval_name}_pair_dataset.csv"

    if not train_path.exists() or not eval_path.exists():
        if verbose:
            print(f"  [{window_label}] SKIP: missing {train_path} or {eval_path}")
        return None

    train_df = load_pair_dataset(train_path)
    eval_df = load_pair_dataset(eval_path)

    # Build sequences
    X_train, y_train, _, _ = build_sequences(
        train_df, FEATURE_COLS, target_col, window_size
    )
    X_val, y_val, val_dates, val_pairs = build_sequences(
        eval_df, FEATURE_COLS, target_col, window_size
    )

    if len(X_train) == 0 or len(X_val) == 0:
        if verbose:
            print(f"  [{window_label}] SKIP: no sequences (train={len(X_train)}, val={len(X_val)})")
        return None

    # Normalize using training stats
    X_train_n, X_val_n, feat_mu, feat_std = normalize_features(X_train, X_val)

    if verbose:
        print(f"  [{window_label}] Training: {len(X_train)} sequences, "
              f"validating: {len(X_val)} sequences")

    # Train
    model, val_mse = train_model(
        X_train_n, y_train, X_val_n, y_val,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    # ── NEW: save model weights + normalisation statistics ─────────────────
    spread_type = "kalman" if "kalman" in target_col else "ols"
    if models_root is None:
        models_root = DEFAULT_CONFIG.processed_dir / "models"
    model_stem = models_root / f"lstm_{spread_type}" / window_label
    save_lstm_model(model, feat_mu, feat_std, model_stem)

    # Predict
    preds = predict(model, X_val_n)

    # Compute metrics
    metrics = compute_metrics(y_val, preds)

    if verbose:
        print(f"  [{window_label}] RMSE={metrics['rmse']:.6f}, "
              f"DW-MSE={metrics['directional_weighted_mse']:.6f}, "
              f"DirAcc={metrics['dir_acc']:.3f}")

    # Build current spread values for deriving predicted_value and predicted_z
    eval_df_sorted = eval_df.sort_values(["pair", "Date"]).reset_index(drop=True)
    current_spreads = []
    current_vols = []
    for pair_name, pair_df in eval_df_sorted.groupby("pair", sort=True):
        pair_df = pair_df.sort_values("Date").reset_index(drop=True)
        n = len(pair_df)
        if n <= window_size:
            continue
        for i in range(window_size, n):
            if spread_col in pair_df.columns:
                current_spreads.append(pair_df[spread_col].iloc[i])
            else:
                current_spreads.append(np.nan)
            if "rolling_vol_20d" in pair_df.columns:
                vol = pair_df["rolling_vol_20d"].iloc[i]
                current_vols.append(vol if not np.isnan(vol) and vol > 1e-8 else 1e-8)
            else:
                current_vols.append(1e-8)

    pred_df = pd.DataFrame({
        "Date": val_dates,
        "pair": val_pairs,
        "predicted_change": preds,
    })

    if len(current_spreads) == len(pred_df):
        pred_df["predicted_value"] = preds + np.array(current_spreads, dtype=np.float32)
        pred_df["predicted_z"] = preds / np.array(current_vols, dtype=np.float32)
    else:
        pred_df["predicted_value"] = np.nan
        pred_df["predicted_z"] = np.nan

    # Save
    out_dir = output_root / f"lstm_{spread_type}" / window_label
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    if verbose:
        print(f"  [{window_label}] Saved {len(pred_df)} predictions -> {out_dir}")

    metrics["window"] = window_label
    metrics["n_train"] = len(X_train)
    metrics["n_val"] = len(X_val)
    return metrics


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_lstm_pipeline(
    spread_types: list[str] | None = None,
    pair_dataset_root: Path | None = None,
    output_root: Path | None = None,
    hidden_size: int | None = None,
    window_size: int | None = None,
    learning_rate: float | None = None,
    do_tune: bool = True,
    target_window: str | None = None,
    run_holdout: bool = False,
    verbose: bool = True,
    models_root: Path | None = None,   # NEW
) -> None:
    """
    Full LSTM pipeline: tune hyperparameters, then predict on all folds.

    Parameters
    ----------
    spread_types : list of "ols" and/or "kalman"
    do_tune : if True, run grid search before prediction.
              if False, use provided hidden_size/window_size/lr.
    target_window : if set, only run this window.
    run_holdout : if True, also run the holdout test set.
    models_root : where to save model .pt files (NEW)
    """
    config = DEFAULT_CONFIG
    pair_root = pair_dataset_root or (config.processed_dir / "pair_datasets")
    out_root = output_root or (config.processed_dir / "predictions")

    if spread_types is None:
        spread_types = ["ols", "kalman"]

    target_map = {
        "ols": ("label_continuous_10d", "spread_ols"),
        "kalman": ("label_kalman_10d", "spread_kalman"),
    }

    for spread_type in spread_types:
        if spread_type not in target_map:
            print(f"Unknown spread type '{spread_type}', skipping.")
            continue

        target_col, spread_col = target_map[spread_type]
        print(f"\n{'='*60}")
        print(f"LSTM — {spread_type.upper()} spread")
        print(f"Target: {target_col}")
        print(f"{'='*60}")

        # Tuning
        if do_tune and hidden_size is None:
            print("\nPhase 1: Hyperparameter tuning")
            best = tune_hyperparameters(pair_root, target_col, verbose=verbose)
            hs = best["hidden_size"]
            ws = best["window_size"]
            lr = best["learning_rate"]

            # Save tuning results
            tune_dir = out_root / f"lstm_{spread_type}"
            tune_dir.mkdir(parents=True, exist_ok=True)
            best["tuning_results"].to_csv(tune_dir / "tuning_results.csv", index=False)
        else:
            hs = hidden_size or 64
            ws = window_size or 20
            lr = learning_rate or 0.001
            print(f"\nUsing fixed hyperparameters: hidden={hs}, window={ws}, lr={lr}")

        print(f"\nFrozen hyperparameters: hidden={hs}, window={ws}, lr={lr}")

        # Phase 2: Predict on each fold
        print("\nPhase 2: Predictions")
        all_metrics = []

        folds_to_run = list(config.expanding_folds)
        if target_window:
            folds_to_run = [f for f in folds_to_run if f.label == target_window]

        for fold in folds_to_run:
            metrics = run_predictions_for_window(
                pair_dataset_root=pair_root,
                window_label=fold.label,
                target_col=target_col,
                spread_col=spread_col,
                hidden_size=hs,
                window_size=ws,
                learning_rate=lr,
                output_root=out_root,
                is_holdout=False,
                verbose=verbose,
                models_root=models_root,   # NEW
            )
            if metrics:
                all_metrics.append(metrics)

        # Phase 3: Holdout (optional)
        if run_holdout:
            holdout_label = config.holdout_split.label
            if target_window is None or target_window == holdout_label:
                print(f"\nPhase 3: Holdout ({holdout_label})")
                metrics = run_predictions_for_window(
                    pair_dataset_root=pair_root,
                    window_label=holdout_label,
                    target_col=target_col,
                    spread_col=spread_col,
                    hidden_size=hs,
                    window_size=ws,
                    learning_rate=lr,
                    output_root=out_root,
                    is_holdout=True,
                    verbose=verbose,
                    models_root=models_root,   # NEW
                )
                if metrics:
                    metrics["is_holdout"] = True
                    all_metrics.append(metrics)

        # Summary
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            summary_dir = out_root / f"lstm_{spread_type}"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_dir / "metrics_summary.csv", index=False)

            print(f"\n{'='*60}")
            print(f"LSTM {spread_type.upper()} Summary:")
            print(f"  Avg val RMSE: {summary_df['rmse'].mean():.6f}")
            if "directional_weighted_mse" in summary_df.columns:
                print(f"  Avg val DW-MSE: {summary_df['directional_weighted_mse'].mean():.6f}")
            print(f"  Avg dir acc: {summary_df['dir_acc'].mean():.3f}")
            if "r2" in summary_df.columns:
                print(f"  Avg R²:      {summary_df['r2'].mean():.4f}")
            if "information_coefficient" in summary_df.columns:
                print(f"  Avg IC:      {summary_df['information_coefficient'].mean():.4f}")
            print(f"  Results: {summary_dir / 'metrics_summary.csv'}")
            print(f"{'='*60}")
        else:
            print(f"\n  No results produced for LSTM {spread_type.upper()}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM spread prediction with hyperparameter tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spread",
        type=str,
        nargs="+",
        default=["ols", "kalman"],
        choices=["ols", "kalman"],
        help="Which spread types to run.",
    )
    parser.add_argument("--window", type=str, default=None, help="Run only this window label.")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden_size (skip tuning).")
    parser.add_argument("--window_size", type=int, default=None, help="Override window_size (skip tuning).")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (skip tuning).")
    parser.add_argument("--no_tune", action="store_true", help="Skip hyperparameter tuning.")
    parser.add_argument("--holdout", action="store_true", help="Also run holdout test set.")
    parser.add_argument("--quiet", action="store_true", help="Minimal output.")
    parser.add_argument(
        "--models_root", type=str, default=None,
        help="Where to save trained .pt model files (default: processed/models/).",
    )
    args = parser.parse_args()

    # If any hyperparameter is manually set, skip tuning
    manual_hp = args.hidden is not None or args.window_size is not None or args.lr is not None
    do_tune = not args.no_tune and not manual_hp

    run_lstm_pipeline(
        spread_types=args.spread,
        hidden_size=args.hidden,
        window_size=args.window_size,
        learning_rate=args.lr,
        do_tune=do_tune,
        target_window=args.window,
        run_holdout=args.holdout,
        verbose=not args.quiet,
        models_root=Path(args.models_root) if args.models_root else None,
    )


if __name__ == "__main__":
    main()
