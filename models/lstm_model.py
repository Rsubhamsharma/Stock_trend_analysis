import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd

from config.settings import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_TIMEOUT_SECONDS,
    LSTM_UNITS,
    PROJECT_ROOT,
    TEST_SIZE,
)
from utils.evaluation import calculate_metrics
from utils.preprocess import scale_series


def create_sequences(scaled_values: np.ndarray, sequence_length: int = LSTM_SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for index in range(sequence_length, len(scaled_values)):
        x_values.append(scaled_values[index - sequence_length:index, 0])
        y_values.append(scaled_values[index, 0])
    return np.array(x_values), np.array(y_values)


def build_lstm_model(sequence_length: int = LSTM_SEQUENCE_LENGTH):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = Sequential(
        [
            Input(shape=(sequence_length, 1)),
            LSTM(LSTM_UNITS, return_sequences=True),
            Dropout(0.2),
            LSTM(LSTM_UNITS),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _train_lstm(data: pd.DataFrame):
    close_values = data["Close"].values.astype(float)
    if len(close_values) <= LSTM_SEQUENCE_LENGTH + 20:
        raise ValueError("At least 80 rows are recommended for LSTM training.")

    scaled_values, scaler = scale_series(close_values)
    x_values, y_values = create_sequences(scaled_values)
    x_values = x_values.reshape((x_values.shape[0], x_values.shape[1], 1))

    split_index = int(len(x_values) * (1 - TEST_SIZE))
    x_train, x_test = x_values[:split_index], x_values[split_index:]
    y_train, y_test = y_values[:split_index], y_values[split_index:]

    model = build_lstm_model()
    model.fit(x_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)
    return model, scaler, x_test, y_test, split_index


def evaluate_lstm_model(data: pd.DataFrame) -> dict:
    model, scaler, x_test, y_test, split_index = _train_lstm(data)
    return _evaluate_trained_lstm(data, model, scaler, x_test, y_test, split_index)


def _evaluate_trained_lstm(
    data: pd.DataFrame,
    model,
    scaler,
    x_test: np.ndarray,
    y_test: np.ndarray,
    split_index: int,
) -> dict:
    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    start = LSTM_SEQUENCE_LENGTH + split_index
    dates = data["Date"].iloc[start:start + len(actual)].reset_index(drop=True)
    actual_df = pd.DataFrame({"Date": dates, "Close": actual})
    predicted_df = pd.DataFrame({"Date": dates, "Predicted_Close": predicted})

    return {
        "metrics": calculate_metrics(actual_df["Close"], predicted_df["Predicted_Close"]),
        "actual": actual_df,
        "predicted": predicted_df,
    }


def _forecast_with_trained_lstm(data: pd.DataFrame, model, scaler, horizon: int) -> pd.DataFrame:
    close_values = data["Close"].values.astype(float)
    scaled_values = scaler.transform(close_values.reshape(-1, 1))
    current_sequence = scaled_values[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, 1)

    predictions = []
    last_date = data["Date"].iloc[-1]

    for _ in range(horizon):
        predicted_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        predicted_close = float(scaler.inverse_transform([[predicted_scaled]])[0, 0])
        next_date = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=1)[0]

        predictions.append({"Date": next_date, "Predicted_Close": predicted_close})
        next_scaled = np.array([[[predicted_scaled]]])
        current_sequence = np.concatenate([current_sequence[:, 1:, :], next_scaled], axis=1)
        last_date = next_date

    return pd.DataFrame(predictions)


def forecast_lstm(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    model, scaler, _, _, _ = _train_lstm(data)
    return _forecast_with_trained_lstm(data, model, scaler, horizon)


def evaluate_and_forecast_lstm_local(data: pd.DataFrame, horizon: int) -> tuple[dict, pd.DataFrame]:
    model, scaler, x_test, y_test, split_index = _train_lstm(data)
    evaluation = _evaluate_trained_lstm(data, model, scaler, x_test, y_test, split_index)
    forecast = _forecast_with_trained_lstm(data, model, scaler, horizon)
    return evaluation, forecast


def _records_to_frame(records: list[dict], value_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["Date"] = pd.to_datetime(frame["Date"])
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def evaluate_and_forecast_lstm(data: pd.DataFrame, horizon: int) -> tuple[dict, pd.DataFrame]:
    with tempfile.TemporaryDirectory(prefix="stock_dashboard_lstm_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.csv"
        output_path = temp_path / "output.json"
        data.to_csv(input_path, index=False)

        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        env["TF_NUM_INTRAOP_THREADS"] = "1"
        env["TF_NUM_INTEROP_THREADS"] = "1"

        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "models.lstm_worker",
            str(input_path),
            str(output_path),
            str(horizon),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=LSTM_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"LSTM training exceeded {LSTM_TIMEOUT_SECONDS} seconds.") from exc

        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            if not details:
                details = (
                    f"worker exited with code {completed.returncode} "
                    f"(hex {completed.returncode & 0xFFFFFFFF:08X})."
                )
            raise RuntimeError(f"LSTM worker failed without closing the dashboard. Details: {details[-1000:]}")

        if not output_path.exists():
            raise RuntimeError("LSTM worker finished but did not return forecast output.")

        with output_path.open("r", encoding="utf-8") as output_file:
            payload = json.load(output_file)

    evaluation = {
        "metrics": payload["evaluation"]["metrics"],
        "actual": _records_to_frame(payload["evaluation"]["actual"], ["Close"]),
        "predicted": _records_to_frame(payload["evaluation"]["predicted"], ["Predicted_Close"]),
    }
    forecast = _records_to_frame(payload["forecast"], ["Predicted_Close"])
    return evaluation, forecast
