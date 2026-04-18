from datetime import date, timedelta

import pandas as pd
import streamlit as st

from config.settings import DEFAULT_FORECAST_HORIZON, DEFAULT_TICKER, MODEL_CACHE_VERSION, SEARCH_DEFAULT_QUERY, SEARCH_MIN_LENGTH
from models.baseline_model import evaluate_baseline_model, forecast_baseline
from models.lstm_model import evaluate_and_forecast_lstm
from services.search_service import SymbolSearchError, search_symbols
from utils.evaluation import build_decision_summary, classify_price_trend, compare_models
from utils.fetch_data import fetch_stock_data, normalize_ticker
from utils.indicators import add_technical_indicators
from utils.preprocess import prepare_price_data
from visualization.charts import (
    create_candlestick_chart,
    create_close_price_chart,
    create_forecast_chart,
    create_indicator_panel,
    create_prediction_comparison_chart,
    create_volume_chart,
)


st.set_page_config(page_title="Stock Market Trend Analysis Dashboard", page_icon=None, layout="wide")


QUICK_RANGES = {
    "1M": 31,
    "3M": 92,
    "6M": 183,
    "1Y": 365,
    "Max": 3650,
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #020617;
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .dashboard-title {
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: 0;
            margin: 0;
        }
        .dashboard-subtitle {
            color: #94a3b8;
            margin-top: 0.2rem;
        }
        .panel {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1.25rem 0 0.55rem 0;
            color: #f8fafc;
        }
        .summary-note {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 1rem;
            color: #dbeafe;
            line-height: 1.6;
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #243044;
            border-radius: 8px;
            padding: 0.75rem;
        }
        @media (max-width: 700px) {
            .dashboard-title {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=900)
def _load_prepared_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    raw_data = fetch_stock_data(ticker, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
    prepared_data = prepare_price_data(raw_data)
    return add_technical_indicators(prepared_data)


@st.cache_data(show_spinner=False, ttl=300)
def _search_symbols_cached(query: str) -> list[dict[str, str]]:
    return search_symbols(query)


def _range_start(range_label: str, end_date: date) -> date:
    return end_date - timedelta(days=QUICK_RANGES[range_label])


def _format_price(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _format_symbol_option(match: dict[str, str]) -> str:
    return f"{match['description']} ({match['symbol']})"


def _symbol_from_option(option: str) -> str:
    if "(" not in option or ")" not in option:
        return normalize_ticker(option)
    return option.rsplit("(", 1)[1].split(")", 1)[0].strip().upper()


def _render_symbol_search(default_ticker: str) -> str | None:
    with st.container(border=True):
        st.caption("Stock search")
        query = st.text_input(
            "Search company or ticker",
            value=default_ticker,
            help="Type a company name or ticker. Examples: app, apple, reliance, AAPL.",
            label_visibility="collapsed",
            placeholder="Search company or ticker",
        )

        cleaned_query = query.strip()
        search_query = cleaned_query if len(cleaned_query) >= SEARCH_MIN_LENGTH else SEARCH_DEFAULT_QUERY
        hint = "Suggestions update after 2+ characters." if len(cleaned_query) < SEARCH_MIN_LENGTH else "Suggestions from Finnhub."

        try:
            matches = _search_symbols_cached(search_query)
        except SymbolSearchError as exc:
            st.warning(str(exc))
            return normalize_ticker(cleaned_query or default_ticker)

        if not matches:
            st.info("No matching symbols found.")
            return normalize_ticker(cleaned_query or default_ticker)

        options = [_format_symbol_option(match) for match in matches]
        selected_option = st.selectbox(
            "Suggestions",
            options,
            label_visibility="collapsed",
            help="Select the symbol to analyze.",
        )
        st.caption(hint)
        return _symbol_from_option(selected_option)


def _card(label: str, value: str, note: str = "") -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value}**")
        if note:
            st.caption(note)


def _render_overview_cards(ticker: str, data: pd.DataFrame, trend: str, best_model: str | None, forecast_horizon: int) -> None:
    latest_close = float(data["Close"].iloc[-1])
    previous_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_close
    recent_change = ((latest_close - previous_close) / previous_close) * 100 if previous_close else 0
    cols = st.columns(6)
    values = [
        ("Ticker", ticker, "Selected symbol"),
        ("Latest Price", _format_price(latest_close), data["Date"].iloc[-1].strftime("%Y-%m-%d")),
        ("Recent Change", _format_pct(recent_change), "Last available session"),
        ("Detected Trend", trend, "SMA, EMA, recent price"),
        ("Best Model", best_model or "Pending", "Selected automatically by RMSE"),
        ("Forecast", f"{forecast_horizon} days", "Business-day horizon"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            _card(label, value, note)


@st.cache_data(show_spinner=False, ttl=3600)
def _run_models(data: pd.DataFrame, forecast_horizon: int, cache_version: str) -> tuple[dict, dict, list[str]]:
    results = {"Baseline": None, "LSTM": None}
    forecasts = {}
    warnings = []

    try:
        results["Baseline"] = evaluate_baseline_model(data)
        forecasts["Baseline"] = forecast_baseline(data, forecast_horizon)
    except Exception as exc:
        warnings.append(f"Baseline model could not be trained: {exc}")

    if len(data) < 80:
        warnings.append("Advanced model unavailable due to insufficient data. Using baseline model.")
    else:
        try:
            results["LSTM"], forecasts["LSTM"] = evaluate_and_forecast_lstm(data, forecast_horizon)
        except Exception as exc:
            warnings.append(_advanced_model_warning(exc))

    return results, forecasts, warnings


def _advanced_model_warning(exc: Exception) -> str:
    message = str(exc)
    if "LSTM worker failed" in message or "tensorflow" in message.lower() or "keras" in message.lower():
        return (
            "Advanced LSTM model is unavailable in this environment, so the dashboard is using the baseline model. "
            "Historical charts, indicators, model comparison, forecast, and summary remain available."
        )
    return f"Advanced LSTM model is unavailable. Using baseline model. Reason: {message}"


def _performance_table(results: dict) -> pd.DataFrame:
    comparison = compare_models(results)
    best_model = comparison["best_model"]
    rows = []
    for model_name in ("Baseline", "LSTM"):
        result = results.get(model_name)
        if result is None:
            rows.append(
                {
                    "Model": model_name,
                    "MAE": "Unavailable",
                    "RMSE": "Unavailable",
                    "MAPE": "Unavailable",
                    "Selection": "Fallback" if model_name == "LSTM" else "Unavailable",
                }
            )
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{metrics['mae']:.4f}",
                "RMSE": f"{metrics['rmse']:.4f}",
                "MAPE": f"{metrics['mape']:.2f}%",
                "Selection": "Best" if model_name == best_model else "Compared",
            }
        )
    return pd.DataFrame(rows)


def _render_prediction_points(summary: dict) -> None:
    interpretation = summary["forecast_interpretation"]
    st.markdown("**Forecast interpretation**")
    point_cols = st.columns(3)
    with point_cols[0]:
        _card("Expected direction", interpretation["direction"], "Forecast endpoint vs latest close")
    with point_cols[1]:
        _card("Expected change", _format_pct(interpretation["expected_change_pct"]), f"Best model: {interpretation['model_used']}")
    with point_cols[2]:
        _card("Model agreement", interpretation["agreement"], "Baseline vs LSTM direction")
    st.markdown(f"- **Why:** {interpretation['why']}")


def _render_model_comparison(results: dict, summary: dict) -> None:
    comparison = summary["model_comparison"]
    best_model = comparison["best_model"] or "Unavailable"
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Best model", best_model)
    with metric_cols[1]:
        st.metric("Selection rule", "Lowest RMSE")
    with metric_cols[2]:
        st.metric("Forecast direction", summary["prediction_direction"])
    with metric_cols[3]:
        st.metric("Confidence", summary["confidence"])
    st.success(f"Best Model: {best_model} ({comparison['reason']})")
    st.dataframe(_performance_table(results), use_container_width=True, hide_index=True)
    st.caption("The future forecast uses only the best available model. RMSE is primary; MAE is used if RMSE is tied.")


def _render_decision_summary(summary: dict) -> None:
    cols = st.columns(5)
    values = [
        ("Trend", summary["trend_direction"], "Price vs SMA/EMA plus recent return"),
        ("RSI", summary["rsi_state"], "Overbought above 70, oversold below 30"),
        ("MACD", summary["macd_state"], "MACD line compared with signal line"),
        ("Forecast", summary["prediction_direction"], "Forecast endpoint compared with latest close"),
        ("Confidence", summary["confidence"], "Based on the best available model error"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            _card(label, value, note)

    st.markdown("**Signal breakdown**")
    for point in summary["key_points"]:
        st.markdown(f"- **{point['title']}**: {point['detail']}")

    st.markdown("**Final note**")
    st.info(summary["explanation"])


def main() -> None:
    _inject_styles()

    st.title("Stock Market Trend Analysis Dashboard")
    st.caption("Chart-first historical analysis, technical indicators, model forecasts, and decision support.")

    with st.container():
        control_cols = st.columns([1.9, 0.75, 0.75])
        with control_cols[0]:
            ticker = _render_symbol_search(DEFAULT_TICKER)
        with control_cols[1]:
            quick_range = st.selectbox("Range", list(QUICK_RANGES.keys()), index=3)
        with control_cols[2]:
            forecast_horizon = st.selectbox(
                "Forecast horizon",
                [7, 14, 30],
                index=[7, 14, 30].index(DEFAULT_FORECAST_HORIZON if DEFAULT_FORECAST_HORIZON in [7, 14, 30] else 7),
            )

    if not ticker:
        st.error("Enter a valid ticker symbol.")
        return

    end_date = date.today()
    start_date = _range_start(quick_range, end_date)

    with st.spinner(f"Loading {ticker} market data..."):
        try:
            indicator_data = _load_prepared_data(ticker, start_date, end_date)
        except Exception as exc:
            st.error(f"Unable to load data for {ticker}: {exc}")
            return

    if indicator_data.empty:
        st.error(f"No market data is available for {ticker}. Check the ticker symbol and try again.")
        return

    results = {"Baseline": None, "LSTM": None}
    forecasts = {}
    model_warnings = []
    with st.spinner("Comparing models and preparing the best forecast..."):
        results, forecasts, model_warnings = _run_models(indicator_data, forecast_horizon, MODEL_CACHE_VERSION)

    for warning in model_warnings:
        st.warning(warning)

    valid_results = {name: result for name, result in results.items() if result is not None}
    preferred_model = compare_models(valid_results)["best_model"]
    summary = build_decision_summary(indicator_data, valid_results, forecasts, preferred_model)

    trend = classify_price_trend(indicator_data)
    _render_overview_cards(ticker, indicator_data, trend, preferred_model, forecast_horizon)

    st.markdown('<div class="section-title">Main Historical Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(create_close_price_chart(indicator_data, ticker), use_container_width=True)

    st.markdown('<div class="section-title">Additional Market Data</div>', unsafe_allow_html=True)
    market_tabs = st.tabs(["Candlestick", "Volume"])
    with market_tabs[0]:
        st.plotly_chart(create_candlestick_chart(indicator_data, ticker), use_container_width=True)
    with market_tabs[1]:
        st.plotly_chart(create_volume_chart(indicator_data, ticker), use_container_width=True)

    st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)
    st.plotly_chart(create_indicator_panel(indicator_data, ticker), use_container_width=True)

    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    if valid_results:
        prediction_tabs = st.tabs(["Model Comparison", "Actual vs Predicted", "Future Forecast"])
        with prediction_tabs[0]:
            _render_model_comparison(results, summary)
        with prediction_tabs[1]:
            st.plotly_chart(create_prediction_comparison_chart(valid_results), use_container_width=True)
        with prediction_tabs[2]:
            st.plotly_chart(create_forecast_chart(indicator_data, forecasts, ticker, preferred_model), use_container_width=True)
            _render_prediction_points(summary)
    else:
        st.warning("Prediction results are unavailable. Historical charts and technical indicators remain available.")

    st.markdown('<div class="section-title">Decision-Support Summary</div>', unsafe_allow_html=True)
    _render_decision_summary(summary)


if __name__ == "__main__":
    main()
