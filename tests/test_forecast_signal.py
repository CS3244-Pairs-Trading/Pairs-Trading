from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.backtest.forecast_signal import ForecastSignal


def _make_price_pair(
    spreads: list[float],
    *,
    beta: float = 1.0,
    start: str = "2016-01-01",
) -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range(start=start, periods=len(spreads))
    log_b = np.full(len(spreads), 4.0)
    log_a = log_b * beta + np.asarray(spreads, dtype=float)
    c1 = pd.Series(np.exp(log_a), index=dates, name="A")
    c2 = pd.Series(np.exp(log_b), index=dates, name="B")
    return c1, c2


def _write_predictions_csv(
    tmpdir: Path,
    window_label: str,
    dates: pd.DatetimeIndex,
    pair: str,
    predicted_values: list[float] | np.ndarray,
    predicted_changes: list[float] | np.ndarray | None = None,
    predicted_z: list[float] | np.ndarray | None = None,
) -> None:
    out_dir = tmpdir / window_label
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_vals = np.asarray(predicted_values, dtype=float)
    pred_changes_arr = (
        np.asarray(predicted_changes, dtype=float)
        if predicted_changes is not None
        else np.zeros(len(dates), dtype=float)
    )
    pred_z_arr = (
        np.asarray(predicted_z, dtype=float)
        if predicted_z is not None
        else np.zeros(len(dates), dtype=float)
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "pair": pair,
            "predicted_change": pred_changes_arr,
            "predicted_value": pred_vals,
            "predicted_z": pred_z_arr,
        }
    )
    df.to_csv(out_dir / "predictions.csv", index=False)


def _make_stats(
    beta: float,
    pair: str = "A|B",
    window: str = "test_window",
) -> dict:
    return {
        "beta": beta,
        "mean": 0.0,
        "std": 1.0,
        "pair": pair,
        "window": window,
    }


def _quantile_thresholds(scores: pd.Series, mode: str, min_obs: int) -> tuple[float, float]:
    scores = scores.dropna()
    pos = scores[scores > 0]
    neg = scores[scores < 0]
    if mode == "decile":
        q_upper, q_lower = 0.90, 0.10
        q_full_upper, q_full_lower = 0.95, 0.05
    else:
        q_upper, q_lower = 0.80, 0.20
        q_full_upper, q_full_lower = 0.80, 0.20

    if len(pos) >= min_obs:
        alpha_l = float(pos.quantile(q_upper))
    elif len(scores) >= min_obs:
        alpha_l = float(scores.quantile(q_full_upper))
    else:
        alpha_l = float(scores.std()) if len(scores) > 1 else 0.01

    if len(neg) >= min_obs:
        alpha_s = float(neg.quantile(q_lower))
    elif len(scores) >= min_obs:
        alpha_s = float(scores.quantile(q_full_lower))
    else:
        alpha_s = -float(scores.std()) if len(scores) > 1 else -0.01

    if not np.isfinite(alpha_l) or alpha_l <= 0:
        alpha_l = max(float(scores.std()), 0.01) if len(scores) > 1 else 0.01
    if not np.isfinite(alpha_s) or alpha_s >= 0:
        alpha_s = -max(float(scores.std()), 0.01) if len(scores) > 1 else -0.01
    return alpha_l, alpha_s


class TestCalibration:
    def test_fit_estimates_robust_spread_floor_from_training(self, tmp_path):
        beta = 1.0
        c1_train, c2_train = _make_price_pair(
            [0.02, 0.05, 0.03, 0.04, 0.06, 0.08, 0.07, 0.09],
            beta=beta,
        )
        dates = pd.bdate_range("2017-01-02", periods=4)
        _write_predictions_csv(tmp_path, "test_window", dates, "A|B", [0.01, 0.02, 0.03, 0.04])

        signal = ForecastSignal(
            tmp_path,
            horizon=1,
            threshold_mode="decile",
            spread_floor_quantile=0.5,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta))

        spread_train = np.log(c1_train) - beta * np.log(c2_train)
        expected_floor = float(spread_train.shift(1).abs().dropna().quantile(0.5))
        assert signal._scale_floor == pytest.approx(expected_floor, rel=1e-6)

    def test_predict_calibrates_thresholds_from_warmup_forecast_scores(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"
        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-01-02",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, -0.02, 0.03, 0.05, -0.01], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=3,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        warmup_scores = pd.Series([1.0, -2.0, 3.0], index=spread_test.index[:3])
        expected_alpha_l, expected_alpha_s = _quantile_thresholds(
            warmup_scores,
            "decile",
            min_obs=1,
        )
        assert signal._alpha_L == pytest.approx(expected_alpha_l, rel=1e-6)
        assert signal._alpha_S == pytest.approx(expected_alpha_s, rel=1e-6)
        assert (signals.iloc[:3] == 0).all()


class TestSignalLogic:
    def test_exits_when_direction_flips(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-02-01",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, -0.02, 0.01], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert list(signals.astype(int)) == [0, 0, 1, 0, 0]

    def test_stale_position_expires_without_refresh(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-03-01",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, 0.01, 0.01], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert list(signals.astype(int)) == [0, 0, 1, 1, 0]

    def test_refreshes_position_on_new_threshold_breach(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-04-03",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, 0.01, 0.05], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert list(signals.astype(int)) == [0, 0, 1, 1, 1]

    def test_missing_predictions_only_carry_until_horizon(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-05-01",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, np.nan, np.nan], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert list(signals.astype(int)) == [0, 0, 1, 1, 0]

    def test_scale_floor_keeps_scores_finite_near_zero_spread(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.20, 0.22, 0.24, 0.26, 0.28, 0.30],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [0.0, 0.0, 0.0, 0.0],
            beta=beta,
            start="2017-06-01",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.001, 0.002, 0.003, -0.001], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=1,
            threshold_mode="quintile",
            warmup_days=2,
            min_calibration_obs=1,
            spread_floor_quantile=0.5,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert np.isfinite(signal._alpha_L)
        assert np.isfinite(signal._alpha_S)
        assert len(signals) == len(c1_test)

    def test_kalman_auto_mode_uses_vol_normalized_scores(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.15, 0.18, 0.20, 0.17, 0.22, 0.21, 0.25, 0.24, 0.27, 0.29, 0.31, 0.30],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [0.28, 0.30, 0.32, 0.29, 0.31],
            beta=beta,
            start="2017-06-15",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, 0.01, -0.01], index=spread_test.index)
        _write_predictions_csv(
            tmp_path / "kalman_model",
            window,
            spread_test.index,
            pair,
            pred_vals.values,
        )

        signal = ForecastSignal(
            tmp_path / "kalman_model",
            horizon=2,
            threshold_mode="decile",
            spread_type="kalman",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signal.predict(c1_test, c2_test)

        assert signal._resolved_score_mode() == "vol_normalized"
        score_series = signal.get_score_series()
        assert score_series is not None
        assert np.isfinite(score_series.dropna()).all()

    def test_reentry_cooldown_blocks_immediate_reentry(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-07-03",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.03, 0.05, -0.01, 0.05, 0.05], index=spread_test.index)
        _write_predictions_csv(tmp_path, window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
            entry_scale=1.2,
            reentry_cooldown_days=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert list(signals.astype(int)) == [0, 0, 1, 0, 0, 1]


class TestPredictionLoading:
    def test_kalman_signal_exposes_dynamic_execution_beta(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"

        c1_train, c2_train = _make_price_pair(
            [0.20, 0.22, 0.18, 0.25, 0.21, 0.27, 0.23, 0.29],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [0.24, 0.26, 0.25, 0.27, 0.28],
            beta=beta,
            start="2017-07-03",
        )
        spread_test = np.log(c1_test) - beta * np.log(c2_test)
        pred_vals = spread_test + pd.Series([0.01, 0.02, 0.03, 0.01, -0.01], index=spread_test.index)
        _write_predictions_csv(tmp_path / "kalman_model", window, spread_test.index, pair, pred_vals.values)

        signal = ForecastSignal(
            tmp_path / "kalman_model",
            horizon=2,
            threshold_mode="decile",
            spread_type="kalman",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signal.predict(c1_test, c2_test)

        beta_exec = signal.get_execution_beta()
        assert isinstance(beta_exec, pd.Series)
        assert len(beta_exec) == len(c1_test)

    def test_missing_predictions_stays_flat(self, tmp_path):
        beta = 1.0
        c1_train, c2_train = _make_price_pair([0.2, 0.3, 0.4, 0.3, 0.5, 0.4], beta=beta)
        c1_test, c2_test = _make_price_pair([0.4, 0.4, 0.4], beta=beta, start="2017-08-01")

        signal = ForecastSignal(tmp_path, horizon=2, threshold_mode="decile")
        signal.fit(c1_train, c2_train, _make_stats(beta))
        signals = signal.predict(c1_test, c2_test)

        assert (signals == 0).all()

    def test_predicted_change_only_files_are_usable(self, tmp_path):
        beta = 1.0
        pair = "A|B"
        window = "test_window"
        c1_train, c2_train = _make_price_pair(
            [0.30, 0.35, 0.40, 0.32, 0.45, 0.38, 0.50, 0.42],
            beta=beta,
        )
        c1_test, c2_test = _make_price_pair(
            [1.0, 1.0, 1.0, 1.0],
            beta=beta,
            start="2017-09-01",
        )
        dates = c1_test.index
        out_dir = tmp_path / window
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Date": dates,
                "pair": pair,
                "predicted_change": [0.01, 0.03, 0.05, -0.01],
                "predicted_value": [np.nan, np.nan, np.nan, np.nan],
            }
        ).to_csv(out_dir / "predictions.csv", index=False)

        signal = ForecastSignal(
            tmp_path,
            horizon=2,
            threshold_mode="decile",
            warmup_days=2,
            min_calibration_obs=1,
        )
        signal.fit(c1_train, c2_train, _make_stats(beta, pair, window))
        signals = signal.predict(c1_test, c2_test)

        assert signal.has_prediction_data
        assert list(signals.astype(int)) == [0, 0, 1, 0]
