from datetime import date, timedelta

import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_START_DATE,
    DEFAULT_TICKER,
    MAX_FORECAST_HORIZON,
)
from models.baseline_model import evaluate_baseline_model, forecast_baseline
from models.lstm_model import evaluate_and_forecast_lstm
from utils.evaluation import build_decision_summary
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


st.set_page_config(
    page_title="Stock Market Trend Analysis Dashboard",
    page_icon=None,
    layout="wide",
)


def _load_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    return fetch_stock_data(normalize_ticker(ticker), start_date.isoformat(), end_date.isoformat())


def _metric_card(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def _render_metrics_table(results: dict) -> None:
    rows = []
    for model_name, result in results.items():
        if result is None:
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "Model": model_name,
                "MAE": round(metrics["mae"], 4),
                "RMSE": round(metrics["rmse"], 4),
                "MAPE (%)": round(metrics["mape"], 2),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Stock Market Trend Analysis Dashboard")
    st.caption("Historical visualization, technical indicators, predictive modeling, and explainable decision support.")

    with st.sidebar:
        st.header("Controls")
        ticker_input = st.text_input(
            "Stock ticker",
            value=DEFAULT_TICKER,
            help="Use exchange symbols such as AAPL, TSLA, MSFT, or NVDA.",
        )
        ticker = normalize_ticker(ticker_input)
        start_date = st.date_input("Start date", value=DEFAULT_START_DATE)
        end_date = st.date_input("End date", value=date.today())
        model_selection = st.selectbox("Model selection", ["Both", "Baseline", "LSTM"])
        forecast_horizon = st.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=MAX_FORECAST_HORIZON,
            value=DEFAULT_FORECAST_HORIZON,
        )
        run_analysis = st.button("Run analysis", type="primary", use_container_width=True)

    if not ticker:
        st.info("Enter a valid stock ticker to begin.")
        return

    if ticker != ticker_input.strip().upper():
        st.sidebar.caption(f"Using ticker symbol: {ticker}")

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    if not run_analysis:
        st.info("Choose a ticker and date range, then run the analysis.")
        return

    with st.spinner("Fetching data and preparing analysis..."):
        try:
            raw_data = _load_data(ticker, start_date, end_date + timedelta(days=1))
            prepared_data = prepare_price_data(raw_data)
            indicator_data = add_technical_indicators(prepared_data)
        except Exception as exc:
            st.error(f"Unable to load data for {ticker}: {exc}")
            return

    if indicator_data.empty or len(indicator_data) < 80:
        st.warning("Not enough historical data for reliable indicators and model evaluation. Try a wider date range.")
        return

    st.subheader(f"{ticker} Historical Market Data")
    price_tab, candle_tab, volume_tab = st.tabs(["Close Price", "Candlestick", "Volume"])

    with price_tab:
        st.plotly_chart(create_close_price_chart(indicator_data, ticker), use_container_width=True)

    with candle_tab:
        st.plotly_chart(create_candlestick_chart(indicator_data, ticker), use_container_width=True)

    with volume_tab:
        st.plotly_chart(create_volume_chart(indicator_data, ticker), use_container_width=True)

    st.subheader("Technical Indicators")
    st.plotly_chart(create_indicator_panel(indicator_data, ticker), use_container_width=True)

    results = {"Baseline": None, "LSTM": None}
    forecasts = {}

    with st.spinner("Training selected model(s) and evaluating forecasts..."):
        if model_selection in ("Both", "Baseline"):
            try:
                baseline_eval = evaluate_baseline_model(indicator_data)
                baseline_future = forecast_baseline(indicator_data, forecast_horizon)
                results["Baseline"] = baseline_eval
                forecasts["Baseline"] = baseline_future
            except Exception as exc:
                st.warning(f"Baseline model could not be trained: {exc}")

        if model_selection in ("Both", "LSTM"):
            try:
                lstm_eval, lstm_future = evaluate_and_forecast_lstm(indicator_data, forecast_horizon)
                results["LSTM"] = lstm_eval
                forecasts["LSTM"] = lstm_future
            except Exception as exc:
                st.warning(f"LSTM model could not be trained: {exc}")

    st.subheader("Model Comparison")
    col1, col2, col3 = st.columns(3)
    valid_results = {name: result for name, result in results.items() if result is not None}
    if valid_results:
        best_model = min(valid_results, key=lambda name: valid_results[name]["metrics"]["rmse"])
        with col1:
            _metric_card("Best model", best_model)
        with col2:
            _metric_card("Latest close", f"${indicator_data['Close'].iloc[-1]:,.2f}")
        with col3:
            _metric_card("Rows analyzed", f"{len(indicator_data):,}")
        _render_metrics_table(valid_results)
    else:
        st.error("No model results are available.")
        return

    comparison_tabs = st.tabs(list(valid_results.keys()))
    for tab, (model_name, result) in zip(comparison_tabs, valid_results.items()):
        with tab:
            st.plotly_chart(
                create_prediction_comparison_chart(result["actual"], result["predicted"], model_name),
                use_container_width=True,
            )

    st.subheader("Forecast")
    st.plotly_chart(create_forecast_chart(indicator_data, forecasts, ticker), use_container_width=True)

    summary = build_decision_summary(indicator_data, valid_results, forecasts)
    st.subheader("Decision Support Summary")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        _metric_card("Trend direction", summary["trend_direction"])
    with summary_cols[1]:
        _metric_card("Model agreement", summary["model_agreement"])
    with summary_cols[2]:
        _metric_card("Confidence", summary["confidence"])
    st.write(summary["explanation"])


if __name__ == "__main__":
    main()
