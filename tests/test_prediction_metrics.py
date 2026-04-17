import numpy as np

from src.models.prediction_metrics import (
    directional_weighted_mse,
    evaluate_regression_predictions,
)


def test_directional_weighted_mse_matches_formula() -> None:
    actual = np.array([1.0, -2.0], dtype=float)
    predicted = np.array([0.5, 1.0], dtype=float)
    gamma = 2.0

    expected = 0.5 * np.mean(
        [
            (1.0 + gamma * 0.0) * (1.0 - 0.5) ** 2,
            (1.0 + gamma * 1.0) * (-2.0 - 1.0) ** 2,
        ]
    )

    assert directional_weighted_mse(actual, predicted, gamma=gamma) == expected


def test_directional_weighted_mse_ignores_gamma_when_signs_match() -> None:
    actual = np.array([1.0, -2.0, 3.0], dtype=float)
    predicted = np.array([0.5, -1.0, 1.5], dtype=float)

    assert directional_weighted_mse(actual, predicted, gamma=0.0) == directional_weighted_mse(
        actual,
        predicted,
        gamma=5.0,
    )


def test_evaluate_regression_predictions_reports_directional_weighted_mse() -> None:
    actual = np.array([1.0, -1.0, 2.0, -2.0], dtype=float)
    predicted_good = np.array([0.9, -0.8, 1.8, -1.9], dtype=float)
    predicted_bad = np.array([0.9, 0.8, 1.8, 1.9], dtype=float)

    good_metrics = evaluate_regression_predictions(actual, predicted_good, gamma=1.5)
    bad_metrics = evaluate_regression_predictions(actual, predicted_bad, gamma=1.5)

    assert "directional_weighted_mse" in good_metrics
    assert bad_metrics["directional_weighted_mse"] > good_metrics["directional_weighted_mse"]
