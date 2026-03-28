import os
import warnings
import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")


def load_spread_data(file_path: str, date_col: str = "Date") -> pd.DataFrame:
    """
    Load spread time series data.

    Expected columns:
    - Date
    - spread

    Optional:
    - pair
    """
    df = pd.read_csv(file_path)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    if "spread" not in df.columns:
        raise ValueError("Input file must contain a 'spread' column.")

    return df


def train_test_split_series(
    series: pd.Series,
    train_ratio: float = 0.8,
):
    """
    Chronological split for time series.
    """
    n = len(series)
    split_idx = int(n * train_ratio)

    train = series.iloc[:split_idx].copy()
    test = series.iloc[split_idx:].copy()

    return train, test


def fit_arma_model(
    train_series: pd.Series,
    p: int,
    q: int,
):
    """
    ARMA(p, q) implemented as ARIMA(p, 0, q).
    """
    model = ARIMA(train_series, order=(p, 0, q))
    fitted_model = model.fit()
    return fitted_model


def rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    p: int,
    q: int,
) -> pd.DataFrame:
    """
    Rolling one-step-ahead forecast.
    Re-fits model each step to avoid leakage.
    """
    history = list(train_series.values)
    predictions = []

    for actual in test_series.values:
        model = ARIMA(history, order=(p, 0, q))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=1)[0]

        predictions.append(forecast)
        history.append(actual)

    results = pd.DataFrame({
        "actual": test_series.values,
        "predicted": predictions
    }, index=test_series.index)

    return results


def evaluate_forecasts(results: pd.DataFrame) -> dict:
    """
    Compute forecast error metrics.
    """
    rmse = np.sqrt(mean_squared_error(results["actual"], results["predicted"]))
    mae = mean_absolute_error(results["actual"], results["predicted"])

    return {
        "rmse": rmse,
        "mae": mae
    }


def generate_trading_signals(
    results: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    Simple signal logic using forecast error z-score.

    error = actual - predicted

    Long spread  => error is very negative
    Short spread => error is very positive
    Exit when error normalizes
    """
    df = results.copy()
    df["forecast_error"] = df["actual"] - df["predicted"]

    error_mean = df["forecast_error"].mean()
    error_std = df["forecast_error"].std()

    if error_std == 0 or np.isnan(error_std):
        df["zscore"] = 0.0
    else:
        df["zscore"] = (df["forecast_error"] - error_mean) / error_std

    position = 0
    signals = []

    for z in df["zscore"]:
        if position == 0:
            if z > entry_z:
                position = -1  # short spread
            elif z < -entry_z:
                position = 1   # long spread
        else:
            if abs(z) < exit_z:
                position = 0

        signals.append(position)

    df["position"] = signals
    return df


def save_outputs(
    output_dir: str,
    forecast_df: pd.DataFrame,
    metrics: dict,
    fitted_model_summary: str,
):
    os.makedirs(output_dir, exist_ok=True)

    forecast_path = os.path.join(output_dir, "arma_forecasts.csv")
    metrics_path = os.path.join(output_dir, "arma_metrics.csv")
    summary_path = os.path.join(output_dir, "arma_model_summary.txt")

    forecast_df.to_csv(forecast_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    with open(summary_path, "w") as f:
        f.write(fitted_model_summary)

    print(f"Saved forecasts to: {forecast_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit ARMA model on spread series.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV containing Date and spread columns."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/arma",
        help="Directory to save outputs."
    )
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="AR order."
    )
    parser.add_argument(
        "--q",
        type=int,
        default=1,
        help="MA order."
    )
    parser.add_argument(
        "--entry_z",
        type=float,
        default=2.0,
        help="Entry z-score threshold."
    )
    parser.add_argument(
        "--exit_z",
        type=float,
        default=0.5,
        help="Exit z-score threshold."
    )

    args = parser.parse_args()

    df = load_spread_data(args.input)

    spread_series = df["spread"].dropna().reset_index(drop=True)

    if len(spread_series) < 30:
        raise ValueError("Spread series is too short for ARMA modeling.")

    train_series, test_series = train_test_split_series(
        spread_series,
        train_ratio=args.train_ratio
    )

    fitted_model = fit_arma_model(train_series, p=args.p, q=args.q)

    forecast_results = rolling_forecast(
        train_series=train_series,
        test_series=test_series,
        p=args.p,
        q=args.q
    )

    metrics = evaluate_forecasts(forecast_results)

    signal_df = generate_trading_signals(
        forecast_results,
        entry_z=args.entry_z,
        exit_z=args.exit_z
    )

    print("\n=== ARMA Forecast Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_outputs(
        output_dir=args.output_dir,
        forecast_df=signal_df,
        metrics=metrics,
        fitted_model_summary=str(fitted_model.summary())
    )


if __name__ == "__main__":
    main()