"""
ForecastSignal -- causal forecast trading with robust spread scaling.

The original Chapter 4 rule trades the predicted percentage change in the
spread. In daily stock pairs that raw percentage can explode when the spread
passes near zero, so this implementation clips the denominator with a
formation-period spread floor and calibrates the entry thresholds from a
causal warmup slice of the model's own forecast scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.pairs_discovery.kalman_hedge import kalman_spread


def _infer_spread_type(
    predictions_root: Path,
    explicit: str | None,
) -> str:
    if explicit is not None:
        if explicit not in {"ols", "kalman"}:
            raise ValueError(f"spread_type must be 'ols' or 'kalman', got '{explicit}'")
        return explicit

    name = predictions_root.name.lower()
    return "kalman" if "kalman" in name else "ols"


def _scaled_change(
    delta: pd.Series,
    base: pd.Series,
    scale_floor: float,
) -> pd.Series:
    safe_base = base.abs().clip(lower=scale_floor)
    return (delta / safe_base) * 100.0


class ForecastSignal:
    """
    Forecast-driven signal with causal threshold calibration.

    Workflow per pair / evaluation period:
    1. Estimate a robust spread floor from the training window.
    2. Convert forecast deltas into clipped percentage spread changes.
    3. Use the first `warmup_days` observations of the test period to fit
       decile/quintile entry thresholds in forecast space.
    4. Trade only after warmup and require forecast renewal every `horizon`
       steps so a stale multi-step forecast cannot persist indefinitely.
    """

    def __init__(
        self,
        predictions_root: str | Path,
        horizon: int = 10,
        threshold_mode: str = "decile",
        spread_type: str | None = None,
        min_abs_spread: float = 1e-6,
        spread_floor_quantile: float = 0.25,
        warmup_days: int = 30,
        min_calibration_obs: int = 10,
    ) -> None:
        if threshold_mode not in ("decile", "quintile"):
            raise ValueError(
                f"threshold_mode must be 'decile' or 'quintile', got '{threshold_mode}'"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got '{horizon}'")
        if not 0.0 <= spread_floor_quantile <= 1.0:
            raise ValueError(
                "spread_floor_quantile must be between 0.0 and 1.0, "
                f"got '{spread_floor_quantile}'"
            )
        if warmup_days < 1:
            raise ValueError(f"warmup_days must be >= 1, got '{warmup_days}'")
        if min_calibration_obs < 1:
            raise ValueError(
                f"min_calibration_obs must be >= 1, got '{min_calibration_obs}'"
            )

        self.predictions_root = Path(predictions_root)
        self.horizon = horizon
        self.threshold_mode = threshold_mode
        self.spread_type = _infer_spread_type(self.predictions_root, spread_type)
        self.min_abs_spread = float(min_abs_spread)
        self.spread_floor_quantile = float(spread_floor_quantile)
        self.warmup_days = int(warmup_days)
        self.min_calibration_obs = int(min_calibration_obs)

        self._alpha_L: float = 0.01
        self._alpha_S: float = -0.01
        self._fallback_alpha_L: float = 0.01
        self._fallback_alpha_S: float = -0.01
        self._beta: float = 1.0
        self._predicted_values: Optional[pd.Series] = None
        self._predicted_changes: Optional[pd.Series] = None
        self._train_c1: Optional[pd.Series] = None
        self._train_c2: Optional[pd.Series] = None
        self._execution_beta: float | pd.Series | None = None
        self._scale_floor: float = self.min_abs_spread

        self._window_cache: dict[str, Optional[pd.DataFrame]] = {}

    @property
    def has_prediction_data(self) -> bool:
        return (
            (self._predicted_values is not None and not self._predicted_values.empty)
            or (self._predicted_changes is not None and not self._predicted_changes.empty)
        )

    def _load_window(self, window_label: str) -> Optional[pd.DataFrame]:
        if window_label in self._window_cache:
            return self._window_cache[window_label]

        path = self.predictions_root / window_label / "predictions.csv"
        if not path.exists():
            self._window_cache[window_label] = None
            return None

        df = pd.read_csv(path, parse_dates=["Date"])
        if "predicted_change" not in df.columns and "predicted_spread_change" in df.columns:
            df = df.rename(columns={"predicted_spread_change": "predicted_change"})
        if "predicted_value" not in df.columns and "predicted_change" not in df.columns:
            raise ValueError(
                f"{path} missing both 'predicted_value' and 'predicted_change'"
            )
        self._window_cache[window_label] = df
        return df

    def has_pair_predictions(self, window_label: str, pair: str) -> bool:
        df = self._load_window(window_label)
        if df is None:
            return False
        sub = df.loc[df["pair"] == pair]
        if sub.empty:
            return False
        has_value = "predicted_value" in sub.columns and bool(sub["predicted_value"].notna().any())
        has_change = "predicted_change" in sub.columns and bool(sub["predicted_change"].notna().any())
        return has_value or has_change

    def get_execution_beta(self) -> float | pd.Series | None:
        return self._execution_beta

    def _build_spread(
        self,
        c1: pd.Series,
        c2: pd.Series,
    ) -> tuple[pd.Series, float | pd.Series]:
        log_c1 = np.log(c1.astype(float))
        log_c2 = np.log(c2.astype(float))

        if self.spread_type == "kalman":
            common = log_c1.index.intersection(log_c2.index)
            spread_arr, beta_arr = kalman_spread(
                log_c1.loc[common].values,
                log_c2.loc[common].values,
            )
            spread = pd.Series(spread_arr, index=common, name="spread")
            beta = pd.Series(beta_arr, index=common, name="beta")
            return spread, beta

        common = log_c1.index.intersection(log_c2.index)
        spread = log_c1.loc[common] - self._beta * log_c2.loc[common]
        return spread.rename("spread"), self._beta

    def _test_spread(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> tuple[pd.Series, float | pd.Series]:
        if self.spread_type != "kalman":
            return self._build_spread(c1_test, c2_test)

        if self._train_c1 is None or self._train_c2 is None:
            return self._build_spread(c1_test, c2_test)

        full_c1 = pd.concat([self._train_c1, c1_test]).sort_index()
        full_c1 = full_c1[~full_c1.index.duplicated(keep="last")]
        full_c2 = pd.concat([self._train_c2, c2_test]).sort_index()
        full_c2 = full_c2[~full_c2.index.duplicated(keep="last")]

        spread_full, beta_full = self._build_spread(full_c1, full_c2)
        test_idx = c1_test.index.intersection(c2_test.index)
        spread = spread_full.reindex(test_idx)
        beta = beta_full.reindex(test_idx) if isinstance(beta_full, pd.Series) else beta_full
        return spread, beta

    def _fit_scale_floor(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
    ) -> float:
        spread, _ = self._build_spread(c1_train, c2_train)
        ref = spread.shift(self.horizon).abs().dropna()
        if ref.empty:
            ref = spread.abs().dropna()
        if ref.empty:
            return self.min_abs_spread
        floor = float(ref.quantile(self.spread_floor_quantile))
        return max(floor, self.min_abs_spread)

    def _thresholds_from_scores(self, scores: pd.Series) -> tuple[float, float]:
        scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            return 0.01, -0.01

        x_pos = scores[scores > 0]
        x_neg = scores[scores < 0]

        if self.threshold_mode == "decile":
            q_upper, q_lower = 0.90, 0.10
            q_full_upper, q_full_lower = 0.95, 0.05
        else:
            q_upper, q_lower = 0.80, 0.20
            q_full_upper, q_full_lower = 0.80, 0.20

        if len(x_pos) >= self.min_calibration_obs:
            alpha_l = float(x_pos.quantile(q_upper))
        elif len(scores) >= self.min_calibration_obs:
            alpha_l = float(scores.quantile(q_full_upper))
        else:
            alpha_l = float(scores.std()) if len(scores) > 1 else 0.01

        if len(x_neg) >= self.min_calibration_obs:
            alpha_s = float(x_neg.quantile(q_lower))
        elif len(scores) >= self.min_calibration_obs:
            alpha_s = float(scores.quantile(q_full_lower))
        else:
            alpha_s = -float(scores.std()) if len(scores) > 1 else -0.01

        if not np.isfinite(alpha_l) or alpha_l <= 0:
            alpha_l = max(float(scores.std()), 0.01) if len(scores) > 1 else 0.01
        if not np.isfinite(alpha_s) or alpha_s >= 0:
            alpha_s = -max(float(scores.std()), 0.01) if len(scores) > 1 else -0.01

        return alpha_l, alpha_s

    def fit(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
        stats: dict,
    ) -> None:
        beta = stats["beta"]
        pair = stats.get("pair")
        window = stats.get("window")

        if pair is None or window is None:
            raise ValueError(
                "ForecastSignal requires 'pair' and 'window' in the stats dict "
                "passed to fit() — make sure the engine injects them."
            )

        self._beta = beta
        self._train_c1 = c1_train.sort_index()
        self._train_c2 = c2_train.sort_index()
        self._execution_beta = beta
        self._scale_floor = self._fit_scale_floor(c1_train, c2_train)

        spread_train, _ = self._build_spread(c1_train, c2_train)
        base_train = spread_train.shift(self.horizon)
        train_scores = _scaled_change(
            spread_train - base_train,
            base_train,
            self._scale_floor,
        )
        self._fallback_alpha_L, self._fallback_alpha_S = self._thresholds_from_scores(
            train_scores
        )
        self._alpha_L = self._fallback_alpha_L
        self._alpha_S = self._fallback_alpha_S

        df = self._load_window(window)
        if df is None:
            self._predicted_values = None
            self._predicted_changes = None
            return

        sub = df.loc[df["pair"] == pair]
        if sub.empty:
            self._predicted_values = None
            self._predicted_changes = None
            return

        if "predicted_value" in sub.columns:
            self._predicted_values = (
                sub.dropna(subset=["predicted_value"])
                .set_index("Date")["predicted_value"]
                .sort_index()
            )
        else:
            self._predicted_values = None

        if "predicted_change" in sub.columns:
            self._predicted_changes = (
                sub.dropna(subset=["predicted_change"])
                .set_index("Date")["predicted_change"]
                .sort_index()
            )
        else:
            self._predicted_changes = None

    def predict(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> pd.Series:
        n = len(c1_test)
        sig_arr = np.zeros(n, dtype=int)

        if (
            (self._predicted_values is None or self._predicted_values.empty)
            and (self._predicted_changes is None or self._predicted_changes.empty)
        ):
            return pd.Series(sig_arr, index=c1_test.index)

        current_spread, beta_exec = self._test_spread(c1_test, c2_test)
        self._execution_beta = beta_exec

        delta_raw = pd.Series(np.nan, index=c1_test.index, dtype=float)
        use_value_series = self._predicted_values is not None and not self._predicted_values.empty
        if use_value_series:
            pred_vals = self._predicted_values.reindex(c1_test.index).astype(float)
            delta_from_value = pred_vals - current_spread
            delta_raw = delta_from_value
        elif self._predicted_changes is not None and not self._predicted_changes.empty:
            delta_from_change = self._predicted_changes.reindex(c1_test.index).astype(float)
            delta_raw = delta_from_change
        delta_score = _scaled_change(delta_raw, current_spread, self._scale_floor).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        warmup_end = min(self.warmup_days, n)
        warmup_scores = delta_score.iloc[:warmup_end]
        if warmup_scores.dropna().shape[0] >= self.min_calibration_obs:
            self._alpha_L, self._alpha_S = self._thresholds_from_scores(warmup_scores)
        else:
            self._alpha_L = self._fallback_alpha_L
            self._alpha_S = self._fallback_alpha_S

        pos = 0
        age = 0

        for i in range(n):
            if i < warmup_end:
                sig_arr[i] = 0
                continue

            raw_i = delta_raw.iloc[i]
            score_i = delta_score.iloc[i]

            if np.isnan(raw_i) or np.isnan(score_i):
                if pos != 0:
                    age += 1
                    if age >= self.horizon:
                        pos = 0
                        age = 0
                sig_arr[i] = pos
                continue

            raw_i = float(raw_i)
            score_i = float(score_i)

            if pos == 0:
                if score_i >= self._alpha_L:
                    pos = 1
                    age = 0
                elif score_i <= self._alpha_S:
                    pos = -1
                    age = 0
            else:
                age += 1

                # Mandatory hold: do NOT exit before the forecast horizon.
                # The model predicts a `horizon`-day change — daily noise in
                # the prediction should not trigger early exits.
                if age < self.horizon:
                    pass  # hold unconditionally
                else:
                    # Reassess at horizon boundary: refresh or exit
                    refreshed = (pos == 1 and score_i >= self._alpha_L) or (
                        pos == -1 and score_i <= self._alpha_S
                    )
                    if refreshed:
                        age = 0  # strong signal persists → hold another horizon
                    else:
                        pos = 0  # signal faded → exit to flat
                        age = 0

            sig_arr[i] = pos

        return pd.Series(sig_arr, index=c1_test.index)

    def __repr__(self) -> str:
        return (
            f"ForecastSignal("
            f"mode={self.threshold_mode}, horizon={self.horizon}, warmup={self.warmup_days}, "
            f"alpha_L={self._alpha_L:.5f}, alpha_S={self._alpha_S:.5f})"
        )
