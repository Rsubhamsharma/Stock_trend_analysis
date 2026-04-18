import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import TEST_SIZE


def prepare_price_data(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"])
    prepared = prepared.sort_values("Date").drop_duplicates(subset=["Date"])

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared[numeric_columns] = prepared[numeric_columns].ffill().bfill()
    prepared = prepared.dropna(subset=numeric_columns)
    return prepared.reset_index(drop=True)


def time_series_train_test_split(data: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(data) * (1 - test_size))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows to create a time-series train/test split.")

    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def scale_series(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    return scaled, scaler
