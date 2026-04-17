from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

DEFAULT_DIRECTIONAL_MSE_GAMMA = 1.0


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def directional_weighted_mse(
    actual: np.ndarray,
    predicted: np.ndarray,
    gamma: float = DEFAULT_DIRECTIONAL_MSE_GAMMA,
) -> float:
    """
    Direction-aware squared-error loss from the referenced paper:

        L(y, y_hat) = 0.5 * [1 + gamma * I(sign(y) != sign(y_hat))] * (y - y_hat)^2

    The returned value is the sample mean of L across observations.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    actual_arr = _as_float_array(actual)
    predicted_arr = _as_float_array(predicted)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if actual_arr.size == 0:
        return float("nan")

    squared_error = (actual_arr - predicted_arr) ** 2
    sign_mismatch = np.sign(actual_arr) != np.sign(predicted_arr)
    penalty = 1.0 + float(gamma) * sign_mismatch.astype(float)
    return float(0.5 * np.mean(penalty * squared_error))


def evaluate_regression_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    gamma: float = DEFAULT_DIRECTIONAL_MSE_GAMMA,
) -> dict[str, float]:
    """Compute shared forecast metrics for spread-change models."""
    actual_arr = _as_float_array(actual)
    predicted_arr = _as_float_array(predicted)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if actual_arr.size == 0:
        return {
            "rmse": float("nan"),
            "r2": float("nan"),
            "information_coefficient": float("nan"),
            "directional_accuracy": float("nan"),
            "profit_weighted_da": float("nan"),
            "directional_weighted_mse": float("nan"),
        }

    diff = predicted_arr - actual_arr
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    var_actual = float(np.var(actual_arr))
    r2 = 1.0 - (mse / var_actual) if var_actual > 1e-16 else float("nan")

    if actual_arr.size >= 3:
        ic, _ = spearmanr(actual_arr, predicted_arr)
        ic = float(ic) if np.isfinite(ic) else float("nan")
    else:
        ic = float("nan")

    nonzero = actual_arr != 0
    if int(nonzero.sum()) > 0:
        sign_correct = np.sign(predicted_arr[nonzero]) == np.sign(actual_arr[nonzero])
        dir_acc = float(np.mean(sign_correct))

        weights = np.abs(actual_arr[nonzero])
        total_weight = float(weights.sum())
        pw_da = float(np.sum(sign_correct * weights) / total_weight) if total_weight > 0 else float("nan")
    else:
        dir_acc = float("nan")
        pw_da = float("nan")

    return {
        "rmse": rmse,
        "r2": r2,
        "information_coefficient": ic,
        "directional_accuracy": dir_acc,
        "profit_weighted_da": pw_da,
        "directional_weighted_mse": directional_weighted_mse(
            actual_arr,
            predicted_arr,
            gamma=gamma,
        ),
    }
