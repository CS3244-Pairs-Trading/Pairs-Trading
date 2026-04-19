"""
run_shap_analysis.py
=====================
Unified SHAP analysis with Datetime fix.
"""

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Set non-interactive backend for server/script environments
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

FEATURE_NAMES = [
    "z_score", "z_score_kalman", "momentum_5d", "momentum_10d",
    "rolling_vol_20d", "rolling_vol_60d", "rolling_corr_60d",
    "days_since_crossing", "kalman_beta", "kalman_beta_change",
    "spread_acceleration",
]

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2] 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path.name}")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out.view(-1, 1) if out.dim() == 1 else out

# --- REVISED ANALYSIS ENGINES ---

def run_xgboost_shap(models_root, data_root, output_dir, spread_types, target_windows, num_samples, instance_plots):
    import shap
    import xgboost as xgb

    for spread_type in spread_types:
        model_dir = models_root / f"xgboost_{spread_type}"
        if not model_dir.exists(): continue
        
        out_sub = _ensure_dir(output_dir / f"xgboost_{spread_type}")
        model_files = sorted(model_dir.glob("*.ubj"))
        if target_windows:
            model_files = [f for f in model_files if f.stem in target_windows]

        for model_path in model_files:
            window_label = model_path.stem
            print(f"\n  [XGBoost {spread_type} | {window_label}]")

            data_folder = data_root / window_label
            target_csv = data_folder / "val_pair_dataset.csv"
            if not target_csv.exists(): target_csv = data_folder / "test_pair_dataset.csv"
            if not target_csv.exists(): continue

            raw_df = pd.read_csv(target_csv)
            # CRITICAL FIX: Convert Date for internal processing
            raw_df["Date"] = pd.to_datetime(raw_df["Date"])
            
            booster = xgb.XGBRegressor()
            booster.load_model(str(model_path))
            
            available_feats = [f for f in FEATURE_NAMES if f in raw_df.columns]
            X = raw_df[available_feats].dropna().values
            X_sample = X[np.random.choice(len(X), min(len(X), num_samples), replace=False)]
            
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=available_feats, plot_type="bar", show=False)
            plt.title(f"XGBoost Importance | {window_label}")
            _save_fig(plt.gcf(), out_sub / f"shap_bar_{window_label}.png")

def run_lstm_shap(models_root, data_root, output_dir, spread_types, target_windows, num_samples, model_type="lstm"):
    import shap
    try:
        if model_type == "lstm":
            from lstm import SpreadLSTM, build_sequences
        else:
            from lstm_encoder_decoder import Seq2SeqSpreadModel, build_seq2seq_samples
    except ImportError as e:
        print(f"  ✗ Import Error: {e}")
        return

    folder_prefix = "lstm_encoder_decoder" if model_type == "lstm_encoder_decoder" else "lstm"

    for spread_type in spread_types:
        model_dir = models_root / f"{folder_prefix}_{spread_type}"
        if not model_dir.exists(): continue

        pt_files = [f for f in model_dir.glob("*.pt") if "_norm" not in f.stem]
        if target_windows:
            pt_files = [f for f in pt_files if f.stem in target_windows]

        out_sub = _ensure_dir(output_dir / f"{folder_prefix}_{spread_type}")

        for pt_path in pt_files:
            window_label = pt_path.stem
            print(f"\n  [{folder_prefix.upper()} {spread_type} | {window_label}]")
            
            norm_file = model_dir / f"{window_label}_norm.npz"
            if not norm_file.exists(): continue
            norm = np.load(str(norm_file))
            
            val_csv = data_root / window_label / "val_pair_dataset.csv"
            if not val_csv.exists(): continue
            
            raw_df = pd.read_csv(val_csv)
            # CRITICAL FIX: Convert Date to datetimelike for .dt accessor in build_sequences
            raw_df["Date"] = pd.to_datetime(raw_df["Date"])
            
            target_col = "label_kalman_10d" if spread_type == "kalman" else "label_continuous_10d"
            available_feats = [f for f in FEATURE_NAMES if f in raw_df.columns]

            state = torch.load(str(pt_path), map_location="cpu")
            h_size = state.get("lstm.weight_hh_l0", state.get("encoder.weight_hh_l0")).shape[1]

            if model_type == "lstm":
                X_seqs, _, _, _ = build_sequences(raw_df, available_feats, target_col, window_size=20)
                model = SpreadLSTM(input_size=len(available_feats), hidden_size=h_size, num_layers=2)
            else:
                samples = build_seq2seq_samples(raw_df, available_feats, f"spread_{spread_type}", target_col, 20, 10)
                X_seqs = samples["X"]
                model = Seq2SeqSpreadModel(input_size=len(available_feats), hidden_size=h_size, horizon=10, num_layers=2)

            model.load_state_dict(state)
            model.eval()

            X_norm = (X_seqs - norm["feat_mu"]) / norm["feat_std"]
            indices = np.random.choice(len(X_norm), min(len(X_norm), num_samples), replace=False)
            X_exp = torch.from_numpy(X_norm[indices].astype(np.float32))
            X_bg = torch.from_numpy(X_norm[np.random.choice(len(X_norm), min(len(X_norm), 50))].astype(np.float32))

            explainer = shap.GradientExplainer(ModelWrapper(model), X_bg)
            shap_v = explainer.shap_values(X_exp)
            if isinstance(shap_v, list): 
                shap_v = shap_v[0]
            # 2. ROBUST DIMENSION COLLAPSING
            # shap_v shape for Seq2Seq: (N_samples, 10_horizon, 20_window, 11_features)
            # shap_v shape for LSTM:    (N_samples, 20_window, 11_features)
            
            # We want to find which axis corresponds to the features (length 11)
            # and average out all other axes.
            num_feats = len(available_feats)
            feat_axis = -1 # Usually the last dimension
            
            for i, dim in enumerate(shap_v.shape):
                if dim == num_feats:
                    feat_axis = i
                    break
            
            # Average every axis EXCEPT the feature axis
            avg_axes = tuple(i for i in range(len(shap_v.shape)) if i != feat_axis)
            feat_shap = np.abs(shap_v).mean(axis=avg_axes)

            # 3. VERIFICATION
            if len(feat_shap) != num_feats:
                print(f"  ✗ Shape Mismatch: Got {len(feat_shap)} values for {num_feats} features. Skipping.")
                continue
            
            plt.figure(figsize=(10, 6))
            plt.barh(available_feats, feat_shap)
            plt.title(f"{model_type.upper()} Global Importance | {window_label}")
            _save_fig(plt.gcf(), out_sub / f"shap_bar_{window_label}.png")

def run_linear_regression_importance(models_root, output_dir, spread_types, target_windows):
    import joblib
    for spread_type in spread_types:
        model_dir = models_root / f"linear_regression_{spread_type}"
        if not model_dir.exists(): continue
        
        out_sub = _ensure_dir(output_dir / f"linear_regression_{spread_type}")
        window_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if target_windows: window_dirs = [d for d in window_dirs if d.name in target_windows]

        for win_dir in window_dirs:
            joblib_files = list(win_dir.glob("*.joblib"))
            if not joblib_files: continue
            
            print(f"\n  [LR {spread_type} | {win_dir.name}]")
            win_coefs = []
            for jf in joblib_files:
                bundle = joblib.load(str(jf))
                win_coefs.append(np.abs(bundle["model"].coef_ * bundle["scaler"].scale_))
            
            mean_imp = np.stack(win_coefs).mean(axis=0)
            plt.figure(figsize=(10, 6))
            plt.barh(bundle["feature_names"], mean_imp)
            plt.title(f"LR Importance | {win_dir.name}")
            _save_fig(plt.gcf(), out_sub / f"coef_bar_{win_dir.name}.png")

# --- CONSOLIDATED MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Unified SHAP Analysis CLI")
    parser.add_argument("--models_root", type=str, default="data/processed/models")
    parser.add_argument("--data_root", type=str, default="data/processed/pair_datasets")
    parser.add_argument("--output_dir", type=str, default="data/processed/shap_outputs")
    parser.add_argument("--model", type=str, default="all", choices=["xgboost", "lstm", "lstm_encoder_decoder", "all"])
    parser.add_argument("--spread", type=str, default="both", choices=["ols", "kalman", "both"])
    parser.add_argument("--window", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--instance_plots", action="store_true")
    args = parser.parse_args()

    models_path = PROJECT_ROOT / args.models_root
    data_path = PROJECT_ROOT / args.data_root
    output_path = _ensure_dir(PROJECT_ROOT / args.output_dir)

    spread_types = ["ols", "kalman"] if args.spread == "both" else [args.spread]
    target_windows = [args.window] if args.window else None

    if args.model in ["all", "xgboost"]:
        print("\n── XGBoost SHAP ──")
        run_xgboost_shap(models_path, data_path, output_path, spread_types, target_windows, args.num_samples, args.instance_plots)

    if args.model in ["all", "lstm"]:
        print("\n── LSTM SHAP ──")
        run_lstm_shap(models_path, data_path, output_path, spread_types, target_windows, args.num_samples, "lstm")

    if args.model in ["all", "lstm_encoder_decoder"]:
        print("\n── Seq2Seq SHAP ──")
        run_lstm_shap(models_path, data_path, output_path, spread_types, target_windows, args.num_samples, "lstm_encoder_decoder")

    print(f"\n✓ SHAP Analysis Complete. Outputs in: {output_path}")

if __name__ == "__main__":
    main()