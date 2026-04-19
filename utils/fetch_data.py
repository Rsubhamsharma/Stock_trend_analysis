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


def _cache_path(ticker: str) -> Path:
    safe_ticker = normalize_ticker(ticker).replace("/", "-")
    return CACHE_DIR / f"{safe_ticker}.csv"


def _read_cached_data(path: Path) -> pd.DataFrame:
    cached = pd.read_csv(path, parse_dates=["Date"])
    return cached.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)


def _filter_date_range(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    filtered = data[(data["Date"] >= start) & (data["Date"] < end)]
    return filtered.sort_values("Date").reset_index(drop=True)


def _cache_covers_range(data: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if data.empty:
        return False
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) - pd.Timedelta(days=1)
    return data["Date"].min() <= start and data["Date"].max() >= end


def _merge_cache(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [incoming]
    if existing is not None and not existing.empty:
        frames.append(existing)
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


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
    path = _cache_path(symbol)

    cached_data = None
    if use_cache and path.exists():
        cached_data = _read_cached_data(path)
        if _cache_covers_range(cached_data, start_date, end_date):
            return _filter_date_range(cached_data, start_date, end_date)

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
        if use_cache and cached_data is not None and not cached_data.empty:
            fallback = _filter_date_range(cached_data, start_date, end_date)
            if not fallback.empty:
                return fallback
        raise ValueError(f"yfinance could not fetch {symbol}. Use an exchange ticker symbol such as AAPL, TSLA, or NVDA.") from exc

    if data.empty:
        if use_cache and cached_data is not None and not cached_data.empty:
            fallback = _filter_date_range(cached_data, start_date, end_date)
            if not fallback.empty:
                return fallback
        raise ValueError(
            f"No market data returned for {symbol}. Enter a valid ticker symbol, not a company name. "
            "For example, use NVDA for NVIDIA."
        )

    data = _clean_downloaded_data(data)
    cache_data = _merge_cache(cached_data, data)
    cache_data.to_csv(path, index=False)
    return _filter_date_range(cache_data, start_date, end_date)
