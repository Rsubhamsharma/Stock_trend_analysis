import pandas as pd

from config.settings import EMA_SPAN, MACD_FAST, MACD_SIGNAL, MACD_SLOW, RSI_WINDOW, SMA_WINDOW


def calculate_sma(series: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(series: pd.Series, span: int = EMA_SPAN) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def calculate_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["SMA"] = calculate_sma(enriched["Close"])
    enriched["EMA"] = calculate_ema(enriched["Close"])
    enriched["RSI"] = calculate_rsi(enriched["Close"])
    enriched["MACD"], enriched["MACD_Signal"], enriched["MACD_Histogram"] = calculate_macd(enriched["Close"])
    return enriched
