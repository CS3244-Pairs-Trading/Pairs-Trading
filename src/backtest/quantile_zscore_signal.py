"""
QuantileZScoreSignal -- adaptive z-score thresholds via empirical quantiles.

Instead of fixed entry/exit thresholds (e.g. +/-2.0), this signal generator
computes thresholds from the training-window z-score distribution. This
accounts for non-Gaussian spread behaviour (fat tails, skew) that makes
fixed thresholds either too aggressive or too conservative.

Usage
-----
    from src.backtest.quantile_zscore_signal import QuantileZScoreSignal

    signal = QuantileZScoreSignal(
        entry_quantile=0.05,   # enter long at 5th pct, short at 95th pct
        exit_quantile=0.40,    # exit when z crosses back to 40th/60th pct
        stop_quantile=0.01,    # stop-loss at 1st/99th pct
    )
    results = engine.run(signal_generator=signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_BACKTEST_PARAMS


def _zscore(
    spread: pd.Series,
    stats: dict,
    use_rolling: bool,
    lookback: int,
) -> pd.Series:
    """Return z-score series using fixed training stats or a rolling window."""
    if use_rolling:
        min_p = max(1, lookback // 2)
        mu = spread.rolling(lookback, min_periods=min_p).mean()
        sig = spread.rolling(lookback, min_periods=min_p).std().clip(lower=1e-8)
    else:
        mu = stats["mean"]
        sig = max(stats["std"], 1e-8)
    return (spread - mu) / sig


class QuantileZScoreSignal:
    """
    Adaptive z-score signal with quantile-calibrated thresholds.

    During fit(), the training-period z-scores are computed and the
    entry/exit/stop thresholds are set from their empirical quantiles.
    This makes thresholds pair-specific and window-specific -- no
    assumption that z-scores are N(0,1).

    The signal logic (enter/exit/reverse/stop) is identical to ZScoreSignal
    so the two are directly comparable.
    """

    def __init__(
        self,
        entry_quantile: float = DEFAULT_BACKTEST_PARAMS.entry_quantile,
        exit_quantile: float = DEFAULT_BACKTEST_PARAMS.exit_quantile,
        stop_quantile: float = DEFAULT_BACKTEST_PARAMS.stop_quantile,
        use_rolling: bool = False,
        rolling_lookback: int = DEFAULT_BACKTEST_PARAMS.rolling_lookback,
        min_obs: int = DEFAULT_BACKTEST_PARAMS.quantile_min_obs,
    ) -> None:
        # Quantile levels (symmetric: lower tail for long, upper for short)
        self.entry_quantile = entry_quantile
        self.exit_quantile = exit_quantile
        self.stop_quantile = stop_quantile

        self.use_rolling = use_rolling
        self.rolling_lookback = rolling_lookback
        self.min_obs = min_obs
        self.entry_exit_gap = DEFAULT_BACKTEST_PARAMS.entry_exit_gap
        self.entry_stop_gap = DEFAULT_BACKTEST_PARAMS.entry_stop_gap

        # Computed during fit(); fallbacks are quantile-specific (independent of ZScoreSignal)
        self._stats: dict = {}
        self.entry_z: float = DEFAULT_BACKTEST_PARAMS.quantile_fallback_entry_z
        self.exit_z: float = DEFAULT_BACKTEST_PARAMS.quantile_fallback_exit_z
        self.stop_z: float = DEFAULT_BACKTEST_PARAMS.quantile_fallback_stop_z

    def fit(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
        stats: dict,
    ) -> None:
        """
        Compute training z-scores and derive thresholds from quantiles.

        The entry threshold is the absolute value at the entry_quantile
        (lower tail). Exit and stop follow the same logic. We take the
        absolute value of the lower-tail quantile so the thresholds work
        symmetrically in the signal loop (same as ZScoreSignal).
        """
        self._stats = stats
        beta = stats["beta"]
        spread = c1_train - beta * c2_train
        z_train = _zscore(spread, stats, self.use_rolling, self.rolling_lookback)
        z_clean = z_train.dropna()

        if len(z_clean) < self.min_obs:
            # Not enough data -- keep fallback defaults
            return

        # Entry: use lower tail quantile, take abs so it works as +/- threshold
        self.entry_z = float(abs(z_clean.quantile(self.entry_quantile)))
        self.exit_z = float(abs(z_clean.quantile(self.exit_quantile)))

        if self.stop_quantile > 0:
            self.stop_z = float(abs(z_clean.quantile(self.stop_quantile)))
        else:
            self.stop_z = 0.0

        # Sanity: entry must be wider than exit
        if self.entry_z <= self.exit_z:
            self.entry_z = self.exit_z + self.entry_exit_gap

        # Sanity: stop must be wider than entry (if enabled)
        if self.stop_z > 0 and self.stop_z <= self.entry_z:
            self.stop_z = self.entry_z + self.entry_stop_gap

    def predict(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> pd.Series:
        """
        Generate signals using quantile-calibrated thresholds.

        Logic is identical to ZScoreSignal.predict() -- only the
        threshold values differ (adaptive vs fixed).
        """
        beta = self._stats["beta"]
        spread = c1_test - beta * c2_test
        z = _zscore(spread, self._stats, self.use_rolling, self.rolling_lookback)

        n = len(c1_test)
        sig_arr = np.zeros(n, dtype=int)
        pos = 0

        for i in range(1, n):
            zi = float(z.iloc[i])
            if np.isnan(zi):
                sig_arr[i] = pos
                continue

            if pos == 0:
                if zi <= -self.entry_z:
                    pos = 1
                elif zi >= self.entry_z:
                    pos = -1
            else:
                if abs(zi) <= self.exit_z:
                    pos = 0
                elif self.stop_z > 0 and abs(zi) >= self.stop_z:
                    pos = 0
                elif pos == 1 and zi >= self.entry_z:
                    pos = -1
                elif pos == -1 and zi <= -self.entry_z:
                    pos = 1

            sig_arr[i] = pos

        return pd.Series(sig_arr, index=c1_test.index)

    def __repr__(self) -> str:
        return (
            f"QuantileZScoreSignal("
            f"entry_q={self.entry_quantile}, exit_q={self.exit_quantile}, "
            f"stop_q={self.stop_quantile}, "
            f"entry_z={self.entry_z:.3f}, exit_z={self.exit_z:.3f}, "
            f"stop_z={self.stop_z:.3f})"
        )