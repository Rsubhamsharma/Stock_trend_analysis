from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import CACHE_DIR


TICKER_ALIASES = {
    "APPL": "AAPL",
    "APPLE": "AAPL",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
}


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    return TICKER_ALIASES.get(cleaned, cleaned)


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    safe_ticker = normalize_ticker(ticker).replace("/", "-")
    return CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _clean_downloaded_data(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {', '.join(missing)}")

    return data[required_columns]


def fetch_stock_data(ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbol = normalize_ticker(ticker)
    path = _cache_path(symbol, start_date, end_date)

    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    try:
        data = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            actions=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        raise ValueError(
            f"yfinance could not fetch {symbol}. Use an exchange ticker symbol such as AAPL, TSLA, or NVDA."
        ) from exc

    if data.empty:
        cached_files = sorted(CACHE_DIR.glob(f"{symbol}_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if use_cache and cached_files:
            return pd.read_csv(cached_files[0], parse_dates=["Date"])
        raise ValueError(
            f"No market data returned for {symbol}. Enter a valid ticker symbol, not a company name. "
            "For example, use NVDA for NVIDIA."
        )

    data = _clean_downloaded_data(data)
    data.to_csv(path, index=False)
    return data
