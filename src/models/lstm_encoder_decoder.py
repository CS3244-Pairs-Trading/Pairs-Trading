"""
LSTM Encoder-Decoder (Seq2Seq) Spread Prediction Model
======================================================

Predicts future spread dynamics from pooled pair data using:
    - LSTM encoder over past feature windows
    - LSTM decoder over future daily spread deltas

For each origin i, decoder target is a length-H sequence:
    [spread(i+1)-spread(i), spread(i+2)-spread(i+1), ..., spread(i+H)-spread(i+H-1)]

Primary evaluation target remains comparable to existing baselines:
    cumulative_change = sum(step_deltas) = spread(i+H) - spread(i)

Outputs:
    data/processed/predictions/lstm_encoder_decoder_<spread_type>/<window>/predictions.csv
    with columns: Date, pair, predicted_change, predicted_value, predicted_z
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_CONFIG
from src.models.prediction_metrics import evaluate_regression_predictions

warnings.filterwarnings("ignore")


FEATURE_COLS: list[str] = [
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

TARGET_MAP: dict[str, tuple[str, str]] = {
    "ols": ("label_continuous_10d", "spread_ols"),
    "kalman": ("label_kalman_10d", "spread_kalman"),
}

HP_GRID: dict[str, list[float | int]] = {
    "hidden_size": [32, 64],
    "window_size": [10, 20],
    "learning_rate": [0.001, 0.0005],
}

# Fixed defaults (unless CLI overrides where supported)
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 7
DEFAULT_HORIZON = 10
DEFAULT_TEACHER_FORCING = 0.5


class Seq2SeqSpreadModel(nn.Module):
    """LSTM encoder-decoder that predicts one future spread delta per decoder step."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        horizon: int,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, n_features)
            target_seq: (batch, horizon) normalized decoder targets (optional)
            teacher_forcing_ratio: probability of feeding ground truth at each decoder step
        Returns:
            pred_seq: (batch, horizon) normalized predicted deltas
        """

        batch_size = x.size(0)
        _, (h, c) = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)
        outputs: list[torch.Tensor] = []

        for step_idx in range(self.horizon):
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            step_pred = self.proj(dec_out.squeeze(1)).squeeze(-1)  # (batch,)
            outputs.append(step_pred)

            use_teacher = (
                self.training
                and target_seq is not None
                and torch.rand(1, device=x.device).item() < teacher_forcing_ratio
            )
            next_input = target_seq[:, step_idx] if use_teacher and target_seq is not None else step_pred
            decoder_input = next_input.unsqueeze(1).unsqueeze(-1)  # (batch, 1, 1)

        return torch.stack(outputs, dim=1)


def load_pair_dataset(path: Path) -> pd.DataFrame | None:
    """Load one pair dataset CSV sorted by pair/date; return None when missing."""
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    if "pair" not in df.columns or "Date" not in df.columns:
        raise ValueError(f"Dataset missing required columns ('pair', 'Date'): {path}")
    return df.dropna(subset=["Date", "pair"]).sort_values(["pair", "Date"]).reset_index(drop=True)


def build_seq2seq_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    spread_col: str,
    target_col: str,
    window_size: int,
    horizon: int,
) -> dict[str, Any]:
    """
    Build pooled seq2seq samples across all pairs.

    A sample at origin index i uses:
      X: rows [i-window_size, ..., i-1] of feature_cols
      y_seq: H daily deltas from spread path starting at i
      y_final: cumulative H-step change at i (label_col when present and finite)
      Date/pair/current_spread/rolling_vol_20d metadata aligned 1:1 with samples
    """

    if df.empty:
        return {
            "X": np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            "y_seq": np.empty((0, horizon), dtype=np.float32),
            "y_final": np.empty((0,), dtype=np.float32),
            "dates": [],
            "pairs": [],
            "current_spread": np.empty((0,), dtype=np.float32),
            "rolling_vol_20d": np.empty((0,), dtype=np.float32),
        }

    required = set(feature_cols) | {"Date", "pair", spread_col, "rolling_vol_20d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for sequence building: {sorted(missing)}")

    has_target = target_col in df.columns
    if not has_target:
        warnings.warn(
            f"Target column '{target_col}' not found; using cumulative path change for y_final.",
            stacklevel=2,
        )

    X_all: list[np.ndarray] = []
    y_seq_all: list[np.ndarray] = []
    y_final_all: list[float] = []
    dates_all: list[str] = []
    pairs_all: list[str] = []
    spread_all: list[float] = []
    vol_all: list[float] = []

    for pair_name, pair_df in df.groupby("pair", sort=True):
        pair_df = pair_df.sort_values("Date").reset_index(drop=True)
        n = len(pair_df)
        if n < window_size + horizon + 1:
            continue

        feat_values = pair_df[feature_cols].to_numpy(dtype=np.float32)
        spread_values = pair_df[spread_col].to_numpy(dtype=np.float32)
        vol_values = pair_df["rolling_vol_20d"].to_numpy(dtype=np.float32)
        if has_target:
            target_values = pair_df[target_col].to_numpy(dtype=np.float32)
        else:
            target_values = np.full(n, np.nan, dtype=np.float32)

        date_values = pair_df["Date"].dt.strftime("%Y-%m-%d").to_numpy()

        for i in range(window_size, n - horizon):
            x_seq = feat_values[i - window_size : i]
            spread_path = spread_values[i : i + horizon + 1]
            step_deltas = np.diff(spread_path)  # len = horizon

            if (
                np.isnan(x_seq).any()
                or np.isnan(spread_path).any()
                or np.isnan(step_deltas).any()
                or len(step_deltas) != horizon
            ):
                continue

            y_cum_from_path = float(step_deltas.sum())
            y_from_label = float(target_values[i]) if has_target else np.nan
            y_final = y_from_label if np.isfinite(y_from_label) else y_cum_from_path
            if not np.isfinite(y_final):
                continue

            current_spread = float(spread_values[i])
            current_vol = float(vol_values[i]) if np.isfinite(vol_values[i]) else np.nan
            if not np.isfinite(current_spread):
                continue

            X_all.append(x_seq)
            y_seq_all.append(step_deltas.astype(np.float32))
            y_final_all.append(y_final)
            dates_all.append(str(date_values[i]))
            pairs_all.append(str(pair_name))
            spread_all.append(current_spread)
            vol_all.append(current_vol)

    if not X_all:
        return {
            "X": np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            "y_seq": np.empty((0, horizon), dtype=np.float32),
            "y_final": np.empty((0,), dtype=np.float32),
            "dates": [],
            "pairs": [],
            "current_spread": np.empty((0,), dtype=np.float32),
            "rolling_vol_20d": np.empty((0,), dtype=np.float32),
        }

    return {
        "X": np.asarray(X_all, dtype=np.float32),
        "y_seq": np.asarray(y_seq_all, dtype=np.float32),
        "y_final": np.asarray(y_final_all, dtype=np.float32),
        "dates": dates_all,
        "pairs": pairs_all,
        "current_spread": np.asarray(spread_all, dtype=np.float32),
        "rolling_vol_20d": np.asarray(vol_all, dtype=np.float32),
    }


def normalize_features(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-normalize input features using training-only statistics."""
    _, _, n_features = X_train.shape
    train_flat = X_train.reshape(-1, n_features)
    mu = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X_train - mu) / std, (X_eval - mu) / std, mu, std


def normalize_targets(
    y_train_seq: np.ndarray,
    y_eval_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize decoder targets per horizon step with train-only statistics."""
    mu = y_train_seq.mean(axis=0)
    std = y_train_seq.std(axis=0)
    std[std < 1e-8] = 1.0
    return (y_train_seq - mu) / std, (y_eval_seq - mu) / std, mu, std


def train_model(
    X_train: np.ndarray,
    y_train_seq: np.ndarray,
    X_val: np.ndarray,
    y_val_seq: np.ndarray,
    hidden_size: int,
    learning_rate: float,
    horizon: int,
    teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = False,
) -> tuple[Seq2SeqSpreadModel, float]:
    """Train seq2seq model with early stopping on validation sequence MSE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqSpreadModel(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        horizon=horizon,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train_seq).to(device),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val_seq).to(device)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            pred_seq = model(
                batch_x,
                target_seq=batch_y,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            loss = loss_fn(pred_seq, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, target_seq=None, teacher_forcing_ratio=0.0)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            avg_train = train_loss_sum / max(n_batches, 1)
            print(
                f"    Epoch {epoch+1:3d}: "
                f"train_seq_mse={avg_train:.6f}, val_seq_mse={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_val_loss


def predict(
    model: Seq2SeqSpreadModel,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Autoregressive inference (no teacher forcing). Returns sequence predictions."""
    device = next(model.parameters()).device
    model.eval()
    out_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            pred = model(batch, target_seq=None, teacher_forcing_ratio=0.0).cpu().numpy()
            out_chunks.append(pred)

    if not out_chunks:
        return np.empty((0, model.horizon), dtype=np.float32)
    return np.vstack(out_chunks).astype(np.float32)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute RMSE, directional accuracy, R², IC, profit-weighted DA, DW-MSE on cumulative change."""
    metrics = evaluate_regression_predictions(actual, predicted)

    return {
        "rmse": metrics["rmse"],
        "dir_acc": metrics["directional_accuracy"],
        "r2": metrics["r2"],
        "information_coefficient": metrics["information_coefficient"],
        "profit_weighted_da": metrics["profit_weighted_da"],
        "directional_weighted_mse": metrics["directional_weighted_mse"],
    }


def tune_hyperparameters(
    pair_dataset_root: Path,
    spread_type: str,
    horizon: int = DEFAULT_HORIZON,
    teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> dict[str, Any]:
    """Grid search over expanding folds; selects by average validation MSE on cumulative H-step change."""
    if spread_type not in TARGET_MAP:
        raise ValueError(f"Unknown spread_type '{spread_type}'. Expected one of {list(TARGET_MAP)}.")
    target_col, spread_col = TARGET_MAP[spread_type]

    grid = list(
        itertools.product(
            HP_GRID["hidden_size"],
            HP_GRID["window_size"],
            HP_GRID["learning_rate"],
        )
    )

    rows: list[dict[str, Any]] = []
    for hidden_size, window_size, learning_rate in grid:
        fold_metrics: list[dict[str, float]] = []
        fold_count = 0
        if verbose:
            print(
                f"\n  Tuning: hidden={hidden_size}, window={window_size}, "
                f"lr={learning_rate}, horizon={horizon}"
            )

        for fold in DEFAULT_CONFIG.expanding_folds:
            train_path = pair_dataset_root / fold.label / "train_pair_dataset.csv"
            val_path = pair_dataset_root / fold.label / "val_pair_dataset.csv"
            train_df = load_pair_dataset(train_path)
            val_df = load_pair_dataset(val_path)
            if train_df is None or val_df is None:
                if verbose:
                    print(f"    Fold {fold.label}: SKIP (missing train/val file)")
                continue

            train_samples = build_seq2seq_samples(
                train_df,
                feature_cols=FEATURE_COLS,
                spread_col=spread_col,
                target_col=target_col,
                window_size=window_size,
                horizon=horizon,
            )
            val_samples = build_seq2seq_samples(
                val_df,
                feature_cols=FEATURE_COLS,
                spread_col=spread_col,
                target_col=target_col,
                window_size=window_size,
                horizon=horizon,
            )

            if len(train_samples["X"]) == 0 or len(val_samples["X"]) == 0:
                if verbose:
                    print(
                        f"    Fold {fold.label}: SKIP (empty sequences: "
                        f"train={len(train_samples['X'])}, val={len(val_samples['X'])})"
                    )
                continue

            X_train, X_val, _, _ = normalize_features(train_samples["X"], val_samples["X"])
            y_train_n, y_val_n, tgt_mu, tgt_std = normalize_targets(
                train_samples["y_seq"], val_samples["y_seq"]
            )

            model, _ = train_model(
                X_train=X_train,
                y_train_seq=y_train_n,
                X_val=X_val,
                y_val_seq=y_val_n,
                hidden_size=int(hidden_size),
                learning_rate=float(learning_rate),
                horizon=horizon,
                teacher_forcing_ratio=teacher_forcing_ratio,
                num_layers=num_layers,
                dropout=dropout,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                verbose=False,
            )

            pred_seq_n = predict(model, X_val, batch_size=batch_size)
            pred_seq = pred_seq_n * tgt_std + tgt_mu
            pred_cum = pred_seq.sum(axis=1)
            actual_cum = val_samples["y_final"]

            metrics = compute_metrics(actual_cum, pred_cum)
            fold_metrics.append(metrics)
            fold_count += 1
            if verbose:
                print(
                    f"    Fold {fold.label}: RMSE={metrics['rmse']:.6f}  "
                    f"R²={metrics['r2']:.4f}  DirAcc={metrics['dir_acc']:.3f}  "
                    f"(train={len(X_train)}, val={len(X_val)})"
                )

        if fold_metrics:
            avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
            avg_dw_mse = float(np.mean([m["directional_weighted_mse"] for m in fold_metrics]))
            avg_r2 = float(np.mean([m["r2"] for m in fold_metrics]))
            avg_dir_acc = float(np.mean([m["dir_acc"] for m in fold_metrics]))
            std_rmse = float(np.std([m["rmse"] for m in fold_metrics]))
        else:
            avg_rmse = float("inf")
            avg_dw_mse = float("inf")
            avg_r2 = float("-inf")
            avg_dir_acc = 0.0
            std_rmse = float("nan")

        composite = 0.5 * avg_r2 + 0.5 * avg_dir_acc

        rows.append(
            {
                "hidden_size": int(hidden_size),
                "window_size": int(window_size),
                "learning_rate": float(learning_rate),
                "horizon": int(horizon),
                "avg_val_rmse": avg_rmse,
                "avg_val_directional_weighted_mse": avg_dw_mse,
                "avg_val_r2": avg_r2,
                "avg_val_dir_acc": avg_dir_acc,
                "composite_score": composite,
                "std_val_rmse": std_rmse,
                "n_folds": int(fold_count),
            }
        )
        if verbose:
            print(f"    Avg RMSE={avg_rmse:.6f}  R²={avg_r2:.4f}  "
                  f"DirAcc={avg_dir_acc:.3f}  Composite={composite:.4f}  "
                  f"({fold_count} folds)")

    results_df = pd.DataFrame(rows).sort_values(
        ["composite_score", "avg_val_directional_weighted_mse", "avg_val_rmse"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_row = results_df.iloc[0]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Seq2Seq tuning results ({spread_type})")
        print(results_df.to_string(index=False))
        print(
            f"\nBest: hidden={int(best_row['hidden_size'])}, "
            f"window={int(best_row['window_size'])}, "
            f"lr={float(best_row['learning_rate'])}, "
            f"DW-MSE={float(best_row['avg_val_directional_weighted_mse']):.6f}, "
            f"R²={float(best_row['avg_val_r2']):.4f}, "
            f"DirAcc={float(best_row['avg_val_dir_acc']):.3f}, "
            f"Composite={float(best_row['composite_score']):.4f}"
        )
        print(f"{'='*60}")

    return {
        "hidden_size": int(best_row["hidden_size"]),
        "window_size": int(best_row["window_size"]),
        "learning_rate": float(best_row["learning_rate"]),
        "avg_val_rmse": float(best_row["avg_val_rmse"]),
        "avg_val_directional_weighted_mse": float(best_row["avg_val_directional_weighted_mse"]),
        "tuning_results": results_df,
    }


def run_predictions_for_window(
    pair_dataset_root: Path,
    window_label: str,
    spread_type: str,
    hidden_size: int,
    window_size: int,
    learning_rate: float,
    output_root: Path,
    horizon: int = DEFAULT_HORIZON,
    teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    is_holdout: bool = False,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """
    Train on window train split and evaluate on val/test split.
    Saves normalized cross-model predictions.csv under lstm_encoder_decoder_<spread_type>/<window>.
    """

    if spread_type not in TARGET_MAP:
        raise ValueError(f"Unknown spread_type '{spread_type}'. Expected one of {list(TARGET_MAP)}.")
    target_col, spread_col = TARGET_MAP[spread_type]

    eval_name = "test" if is_holdout else "val"
    train_path = pair_dataset_root / window_label / "train_pair_dataset.csv"
    eval_path = pair_dataset_root / window_label / f"{eval_name}_pair_dataset.csv"

    train_df = load_pair_dataset(train_path)
    eval_df = load_pair_dataset(eval_path)
    if train_df is None or eval_df is None:
        if verbose:
            print(f"  [{window_label}] SKIP: missing {train_path.name} or {eval_path.name}")
        return None

    train_samples = build_seq2seq_samples(
        train_df,
        feature_cols=FEATURE_COLS,
        spread_col=spread_col,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
    )
    eval_samples = build_seq2seq_samples(
        eval_df,
        feature_cols=FEATURE_COLS,
        spread_col=spread_col,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
    )

    if len(train_samples["X"]) == 0 or len(eval_samples["X"]) == 0:
        if verbose:
            print(
                f"  [{window_label}] SKIP: no sequences "
                f"(train={len(train_samples['X'])}, {eval_name}={len(eval_samples['X'])})"
            )
        return None

    X_train, X_eval, _, _ = normalize_features(train_samples["X"], eval_samples["X"])
    y_train_n, y_eval_n, tgt_mu, tgt_std = normalize_targets(
        train_samples["y_seq"], eval_samples["y_seq"]
    )

    if verbose:
        print(
            f"  [{window_label}] Training seq2seq: train={len(X_train)}, "
            f"{eval_name}={len(X_eval)}, horizon={horizon}"
        )

    model, _ = train_model(
        X_train=X_train,
        y_train_seq=y_train_n,
        X_val=X_eval,
        y_val_seq=y_eval_n,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        horizon=horizon,
        teacher_forcing_ratio=teacher_forcing_ratio,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
    )

    pred_seq_n = predict(model, X_eval, batch_size=batch_size)
    pred_seq = pred_seq_n * tgt_std + tgt_mu
    pred_cum = pred_seq.sum(axis=1).astype(np.float32)
    actual_cum = eval_samples["y_final"].astype(np.float32)

    current_spread = eval_samples["current_spread"].astype(np.float32)
    rolling_vol = eval_samples["rolling_vol_20d"].astype(np.float32)
    safe_vol = np.where(np.isfinite(rolling_vol) & (rolling_vol > 1e-8), rolling_vol, 1e-8)

    pred_df = pd.DataFrame(
        {
            "Date": eval_samples["dates"],
            "pair": eval_samples["pairs"],
            "predicted_change": pred_cum,
            "predicted_value": current_spread + pred_cum,
            "predicted_z": pred_cum / safe_vol,
        }
    )
    pred_df = pred_df.sort_values(["pair", "Date"]).drop_duplicates(
        subset=["Date", "pair"], keep="first"
    )

    out_dir = output_root / f"lstm_encoder_decoder_{spread_type}" / window_label
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = compute_metrics(actual_cum, pred_cum)
    if verbose:
        print(
            f"  [{window_label}] RMSE={metrics['rmse']:.6f}, "
            f"DW-MSE={metrics['directional_weighted_mse']:.6f}, "
            f"DirAcc={metrics['dir_acc']:.3f}"
        )
        print(f"  [{window_label}] Saved predictions -> {out_dir / 'predictions.csv'}")

    row: dict[str, Any] = {
        "window": window_label,
        "rmse": metrics["rmse"],
        "directional_weighted_mse": metrics["directional_weighted_mse"],
        "dir_acc": metrics["dir_acc"],
        "n_train": int(len(X_train)),
        "hidden_size": int(hidden_size),
        "window_size": int(window_size),
        "learning_rate": float(learning_rate),
        "horizon": int(horizon),
        "is_holdout": bool(is_holdout),
    }
    row[f"n_{eval_name}"] = int(len(X_eval))
    return row


def run_lstm_encoder_decoder_pipeline(
    spread_types: list[str] | None = None,
    pair_dataset_root: Path | None = None,
    output_root: Path | None = None,
    hidden_size: int | None = None,
    window_size: int | None = None,
    learning_rate: float | None = None,
    do_tune: bool = True,
    target_window: str | None = None,
    run_holdout: bool = False,
    horizon: int = DEFAULT_HORIZON,
    teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> None:
    """Run tuning + per-window prediction pipeline for one or both spread variants."""
    pair_root = pair_dataset_root or (DEFAULT_CONFIG.processed_dir / "pair_datasets")
    out_root = output_root or (DEFAULT_CONFIG.processed_dir / "predictions")
    spreads = spread_types or ["ols", "kalman"]

    for spread_type in spreads:
        if spread_type not in TARGET_MAP:
            print(f"Unknown spread type '{spread_type}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"LSTM Encoder-Decoder — {spread_type.upper()}")
        print(f"{'='*60}")

        manual_hp = (
            hidden_size is not None or window_size is not None or learning_rate is not None
        )
        should_tune = do_tune and not manual_hp

        if should_tune:
            print("\nPhase 1: Hyperparameter tuning")
            best = tune_hyperparameters(
                pair_dataset_root=pair_root,
                spread_type=spread_type,
                horizon=horizon,
                teacher_forcing_ratio=teacher_forcing_ratio,
                num_layers=num_layers,
                dropout=dropout,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                verbose=verbose,
            )
            hs = int(best["hidden_size"])
            ws = int(best["window_size"])
            lr = float(best["learning_rate"])
            model_dir = out_root / f"lstm_encoder_decoder_{spread_type}"
            model_dir.mkdir(parents=True, exist_ok=True)
            best["tuning_results"].to_csv(model_dir / "tuning_results.csv", index=False)
        else:
            hs = int(hidden_size if hidden_size is not None else 64)
            ws = int(window_size if window_size is not None else 20)
            lr = float(learning_rate if learning_rate is not None else 0.001)
            print(f"\nUsing fixed hyperparameters: hidden={hs}, window={ws}, lr={lr}")

        print(
            f"\nFrozen hyperparameters: hidden={hs}, window={ws}, lr={lr}, "
            f"horizon={horizon}, tf={teacher_forcing_ratio}"
        )

        all_metrics: list[dict[str, Any]] = []

        print("\nPhase 2: Validation windows")
        folds = list(DEFAULT_CONFIG.expanding_folds)
        if target_window is not None:
            folds = [f for f in folds if f.label == target_window]
            if not folds:
                print(f"  No matching validation fold for window='{target_window}'.")

        for fold in folds:
            metrics_row = run_predictions_for_window(
                pair_dataset_root=pair_root,
                window_label=fold.label,
                spread_type=spread_type,
                hidden_size=hs,
                window_size=ws,
                learning_rate=lr,
                output_root=out_root,
                horizon=horizon,
                teacher_forcing_ratio=teacher_forcing_ratio,
                num_layers=num_layers,
                dropout=dropout,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                is_holdout=False,
                verbose=verbose,
            )
            if metrics_row is not None:
                all_metrics.append(metrics_row)

        if run_holdout:
            holdout_label = DEFAULT_CONFIG.holdout_split.label
            if target_window is None or target_window == holdout_label:
                print(f"\nPhase 3: Holdout ({holdout_label})")
                metrics_row = run_predictions_for_window(
                    pair_dataset_root=pair_root,
                    window_label=holdout_label,
                    spread_type=spread_type,
                    hidden_size=hs,
                    window_size=ws,
                    learning_rate=lr,
                    output_root=out_root,
                    horizon=horizon,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    is_holdout=True,
                    verbose=verbose,
                )
                if metrics_row is not None:
                    all_metrics.append(metrics_row)

        model_dir = out_root / f"lstm_encoder_decoder_{spread_type}"
        model_dir.mkdir(parents=True, exist_ok=True)
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            summary_df.to_csv(model_dir / "metrics_summary.csv", index=False)
            print(f"\nSummary saved: {model_dir / 'metrics_summary.csv'}")
            print(
                f"Avg RMSE={summary_df['rmse'].mean():.6f}, "
                f"Avg DW-MSE={summary_df['directional_weighted_mse'].mean():.6f}, "
                f"Avg DirAcc={summary_df['dir_acc'].mean():.3f}"
            )
        else:
            print("  No results produced.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM encoder-decoder spread prediction with fold-based tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spread",
        type=str,
        nargs="+",
        default=["ols", "kalman"],
        choices=["ols", "kalman"],
        help="Spread types to run.",
    )
    parser.add_argument("--window", type=str, default=None, help="Run only this window label.")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden_size (skip tuning).")
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override input window size (skip tuning).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (skip tuning).")
    parser.add_argument("--no_tune", action="store_true", help="Skip hyperparameter tuning.")
    parser.add_argument("--holdout", action="store_true", help="Also run holdout test split.")
    parser.add_argument("--quiet", action="store_true", help="Minimal output.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon H.")
    parser.add_argument(
        "--teacher_forcing",
        type=float,
        default=DEFAULT_TEACHER_FORCING,
        help="Teacher forcing ratio in [0, 1].",
    )
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0")
    if args.teacher_forcing < 0.0 or args.teacher_forcing > 1.0:
        raise ValueError("--teacher_forcing must be in [0, 1]")

    manual_hp = args.hidden is not None or args.window_size is not None or args.lr is not None
    do_tune = not args.no_tune and not manual_hp

    run_lstm_encoder_decoder_pipeline(
        spread_types=args.spread,
        hidden_size=args.hidden,
        window_size=args.window_size,
        learning_rate=args.lr,
        do_tune=do_tune,
        target_window=args.window,
        run_holdout=args.holdout,
        horizon=args.horizon,
        teacher_forcing_ratio=args.teacher_forcing,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
