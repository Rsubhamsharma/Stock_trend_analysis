import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.settings import CONFIDENCE_HIGH_RMSE_PCT, CONFIDENCE_MEDIUM_RMSE_PCT


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


def confidence_from_rmse(rmse: float, latest_close: float) -> tuple[str, float]:
    if latest_close <= 0 or not np.isfinite(rmse):
        return "Low", float("inf")
    rmse_pct = (rmse / latest_close) * 100
    if rmse_pct <= CONFIDENCE_HIGH_RMSE_PCT:
        return "High", rmse_pct
    if rmse_pct <= CONFIDENCE_MEDIUM_RMSE_PCT:
        return "Medium", rmse_pct
    return "Low", rmse_pct


def get_best_model(results: dict) -> str | None:
    valid_results = {name: result for name, result in results.items() if result is not None}
    if not valid_results:
        return None
    return min(valid_results, key=lambda name: (valid_results[name]["metrics"]["rmse"], valid_results[name]["metrics"]["mae"]))


def classify_price_trend(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    recent_return = ((data["Close"].iloc[-1] - data["Close"].iloc[-20]) / data["Close"].iloc[-20]) * 100 if len(data) >= 20 else 0
    above_averages = latest["Close"] > latest["SMA"] and latest["Close"] > latest["EMA"]
    below_averages = latest["Close"] < latest["SMA"] and latest["Close"] < latest["EMA"]

    if above_averages and recent_return > 1:
        return "Upward"
    if below_averages and recent_return < -1:
        return "Downward"
    return "Neutral"


def classify_rsi(rsi_value: float) -> str:
    if rsi_value >= 70:
        return "Overbought"
    if rsi_value <= 30:
        return "Oversold"
    return "Neutral"


def classify_macd(latest_macd: float, latest_signal: float, previous_macd: float | None = None, previous_signal: float | None = None) -> str:
    if previous_macd is not None and previous_signal is not None:
        if previous_macd <= previous_signal and latest_macd > latest_signal:
            return "Bullish crossover"
        if previous_macd >= previous_signal and latest_macd < latest_signal:
            return "Bearish crossover"
    if latest_macd > latest_signal:
        return "Bullish momentum"
    if latest_macd < latest_signal:
        return "Bearish momentum"
    return "Neutral momentum"


def forecast_direction(latest_close: float, forecast: pd.DataFrame | None) -> tuple[str, float | None]:
    if forecast is None or forecast.empty:
        return "Unavailable", None
    final_forecast = float(forecast["Predicted_Close"].iloc[-1])
    change_pct = ((final_forecast - latest_close) / latest_close) * 100
    if change_pct > 1:
        return "Upward", change_pct
    if change_pct < -1:
        return "Downward", change_pct
    return "Neutral", change_pct


def compare_model_directions(data: pd.DataFrame, forecasts: dict) -> str:
    latest_close = float(data["Close"].iloc[-1])
    directions = [
        forecast_direction(latest_close, forecast)[0]
        for forecast in forecasts.values()
        if forecast is not None and not forecast.empty
    ]
    if len(directions) < 2:
        return "Unavailable"
    return "Agree" if len(set(directions)) == 1 else "Disagree"


def compare_models(results: dict) -> dict[str, str | float | None]:
    best_model = get_best_model(results)
    if best_model is None:
        return {
            "best_model": None,
            "reason": "No model results are available.",
            "baseline_rmse": None,
            "lstm_rmse": None,
        }

    best_metrics = results[best_model]["metrics"]
    reason = f"{best_model} has the lowest RMSE ({best_metrics['rmse']:.4f}); MAE is used as a tie-breaker."
    return {
        "best_model": best_model,
        "reason": reason,
        "baseline_rmse": results["Baseline"]["metrics"]["rmse"] if results.get("Baseline") else None,
        "lstm_rmse": results["LSTM"]["metrics"]["rmse"] if results.get("LSTM") else None,
    }


def build_decision_summary(data: pd.DataFrame, results: dict, forecasts: dict, preferred_model: str | None = None) -> dict[str, str | float | None]:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    latest_close = float(latest["Close"])
    sma = float(latest["SMA"])
    ema = float(latest["EMA"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    best_model = get_best_model(results)
    model_for_forecast = preferred_model if preferred_model in forecasts else best_model
    selected_forecast = forecasts.get(model_for_forecast) if model_for_forecast else None
    prediction_direction, expected_change = forecast_direction(latest_close, selected_forecast)
    best_mape = min((result["metrics"]["mape"] for result in results.values() if result is not None), default=float("inf"))
    best_rmse = min((result["metrics"]["rmse"] for result in results.values() if result is not None), default=float("inf"))
    confidence, rmse_pct = confidence_from_rmse(best_rmse, latest_close)

    trend_direction = classify_price_trend(data)
    rsi_state = classify_rsi(float(latest["RSI"]))
    macd_state = classify_macd(
        float(latest["MACD"]),
        float(latest["MACD_Signal"]),
        float(previous["MACD"]),
        float(previous["MACD_Signal"]),
    )
    agreement = compare_model_directions(data, forecasts)

    average_context = "above" if latest_close > sma and latest_close > ema else "below" if latest_close < sma and latest_close < ema else "near"
    change_text = "not available" if expected_change is None else f"{expected_change:+.2f}%"
    accuracy_text = "limited" if not np.isfinite(best_mape) else f"{best_mape:.2f}% MAPE"
    rmse_text = "unavailable" if not np.isfinite(best_rmse) else f"{best_rmse:.4f} RMSE ({rmse_pct:.2f}% of latest price)"
    model_comparison = compare_models(results)

    key_points = [
        {
            "title": f"Trend is {trend_direction}",
            "detail": (
                f"Latest close is ${latest_close:,.2f}, SMA is ${sma:,.2f}, and EMA is ${ema:,.2f}. "
                f"The trend label comes from price position versus SMA/EMA plus recent price movement."
            ),
        },
        {
            "title": f"RSI is {rsi_state}",
            "detail": (
                f"Current RSI is {rsi:.2f}. Values above 70 are treated as overbought, below 30 as oversold, "
                "and the middle range as neutral."
            ),
        },
        {
            "title": macd_state,
            "detail": (
                f"MACD is {macd:.4f} and the signal line is {macd_signal:.4f}. "
                "MACD above the signal line indicates bullish momentum; below it indicates bearish momentum."
            ),
        },
        {
            "title": f"Forecast points {prediction_direction}",
            "detail": (
                f"The {model_for_forecast or 'selected'} forecast endpoint implies {change_text} over the selected horizon. "
                "Forecast direction is based on the predicted endpoint compared with the latest close."
            ),
        },
        {
            "title": f"Confidence is {confidence}",
            "detail": (
                f"Confidence is derived from the best model RMSE: {rmse_text}. "
                f"Thresholds are configurable: high <= {CONFIDENCE_HIGH_RMSE_PCT:.1f}% and medium <= {CONFIDENCE_MEDIUM_RMSE_PCT:.1f}% of the latest price."
            ),
        },
    ]

    forecast_interpretation = {
        "direction": prediction_direction,
        "expected_change_pct": expected_change,
        "agreement": agreement,
        "model_used": model_for_forecast or "Unavailable",
        "why": (
            f"Compared the latest close (${latest_close:,.2f}) with the forecast endpoint. "
            f"The forecast uses {model_for_forecast or 'the available model'} because it had the lowest RMSE. "
            f"Baseline/LSTM agreement is {agreement.lower()} because "
            "both available forecast directions match." if agreement == "Agree" else
            f"Compared the latest close (${latest_close:,.2f}) with the forecast endpoint. "
            f"The forecast uses {model_for_forecast or 'the available model'} because it had the lowest RMSE. "
            f"Baseline/LSTM agreement is {agreement.lower()} because fewer than two comparable forecasts are available or their directions differ."
        ),
    }

    explanation = (
        f"{trend_direction} trend: price is {average_context} the SMA/EMA reference zone. "
        f"RSI is {rsi_state.lower()}, and MACD shows {macd_state.lower()}. "
        f"The automatically selected {model_for_forecast or 'available'} model forecast suggests {prediction_direction.lower()} movement "
        f"over the horizon ({change_text}). Model agreement is {agreement.lower()} with {confidence.lower()} confidence "
        f"based on {rmse_text}. This is decision support, not financial advice."
    )

    return {
        "trend_direction": trend_direction,
        "rsi_state": rsi_state,
        "macd_state": macd_state,
        "prediction_direction": prediction_direction,
        "expected_change_pct": expected_change,
        "model_agreement": agreement,
        "confidence": confidence,
        "best_model": best_model,
        "model_for_forecast": model_for_forecast,
        "model_comparison": model_comparison,
        "rmse_pct": rmse_pct,
        "key_points": key_points,
        "forecast_interpretation": forecast_interpretation,
        "explanation": explanation,
    }
