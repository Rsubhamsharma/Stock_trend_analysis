import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    nonzero_mask = actual_values != 0
    mape = np.mean(np.abs((actual_values[nonzero_mask] - predicted_values[nonzero_mask]) / actual_values[nonzero_mask])) * 100

    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape": float(mape),
    }


def confidence_from_mape(mape: float) -> str:
    if mape <= 3:
        return "High"
    if mape <= 7:
        return "Medium"
    return "Low"


def build_decision_summary(data: pd.DataFrame, results: dict, forecasts: dict) -> dict[str, str]:
    latest_close = float(data["Close"].iloc[-1])
    forecast_directions = {}

    for model_name, forecast in forecasts.items():
        if forecast is None or forecast.empty:
            continue
        final_forecast = float(forecast["Predicted_Close"].iloc[-1])
        change_pct = ((final_forecast - latest_close) / latest_close) * 100
        if change_pct > 1:
            forecast_directions[model_name] = "Upward"
        elif change_pct < -1:
            forecast_directions[model_name] = "Downward"
        else:
            forecast_directions[model_name] = "Neutral"

    if not forecast_directions:
        trend_direction = "Neutral"
        model_agreement = "Unavailable"
    else:
        direction_counts = pd.Series(forecast_directions).value_counts()
        trend_direction = str(direction_counts.idxmax())
        model_agreement = "Agree" if direction_counts.iloc[0] == len(forecast_directions) else "Mixed"

    best_mape = min(result["metrics"]["mape"] for result in results.values())
    confidence = confidence_from_mape(best_mape)

    explanation = (
        f"The summary compares the latest close with the forecast endpoint. "
        f"Confidence is based on the best model MAPE ({best_mape:.2f}%). "
        f"This is decision support, not financial advice."
    )

    return {
        "trend_direction": trend_direction,
        "model_agreement": model_agreement,
        "confidence": confidence,
        "explanation": explanation,
    }
