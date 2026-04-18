import pandas as pd
from sklearn.linear_model import LinearRegression

from config.settings import BASELINE_WINDOW
from utils.evaluation import calculate_metrics
from utils.preprocess import time_series_train_test_split


def _build_features(data: pd.DataFrame, window: int = BASELINE_WINDOW) -> pd.DataFrame:
    features = data[["Date", "Close"]].copy()
    features["Day_Index"] = range(len(features))
    features["Rolling_Mean"] = features["Close"].rolling(window=window, min_periods=1).mean()
    features["Lag_1"] = features["Close"].shift(1)
    features["Lag_2"] = features["Close"].shift(2)
    features = features.dropna().reset_index(drop=True)
    return features


def train_baseline_model(data: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame]:
    features = _build_features(data)
    train, _ = time_series_train_test_split(features)
    model = LinearRegression()
    model.fit(train[["Day_Index", "Rolling_Mean", "Lag_1", "Lag_2"]], train["Close"])
    return model, features


def evaluate_baseline_model(data: pd.DataFrame) -> dict:
    model, features = train_baseline_model(data)
    _, test = time_series_train_test_split(features)
    predicted = model.predict(test[["Day_Index", "Rolling_Mean", "Lag_1", "Lag_2"]])

    actual = pd.DataFrame({"Date": test["Date"], "Close": test["Close"]}).reset_index(drop=True)
    prediction = pd.DataFrame({"Date": test["Date"], "Predicted_Close": predicted}).reset_index(drop=True)

    return {
        "metrics": calculate_metrics(actual["Close"], prediction["Predicted_Close"]),
        "actual": actual,
        "predicted": prediction,
    }


def forecast_baseline(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    model, features = train_baseline_model(data)
    history = data[["Date", "Close"]].copy().reset_index(drop=True)
    predictions = []

    for _ in range(horizon):
        next_date = pd.bdate_range(history["Date"].iloc[-1] + pd.Timedelta(days=1), periods=1)[0]
        day_index = len(features) + len(predictions)
        rolling_mean = history["Close"].tail(BASELINE_WINDOW).mean()
        lag_1 = history["Close"].iloc[-1]
        lag_2 = history["Close"].iloc[-2]
        next_features = pd.DataFrame(
            [[day_index, rolling_mean, lag_1, lag_2]],
            columns=["Day_Index", "Rolling_Mean", "Lag_1", "Lag_2"],
        )
        predicted_close = float(model.predict(next_features)[0])

        predictions.append({"Date": next_date, "Predicted_Close": predicted_close})
        history.loc[len(history)] = {"Date": next_date, "Close": predicted_close}

    return pd.DataFrame(predictions)
