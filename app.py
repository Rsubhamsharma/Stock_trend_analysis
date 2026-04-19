from datetime import date, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.settings import DEFAULT_FORECAST_HORIZON, DEFAULT_TICKER, FINNHUB_API_KEY, MODEL_CACHE_VERSION, SEARCH_DEFAULT_QUERY, SEARCH_MIN_LENGTH
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

PLOTLY_CHART_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["zoom2d", "select2d", "lasso2d"],
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #020617;
            color: #e5e7eb;
        }
        #MainMenu,
        footer,
        header,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        .block-container {
            padding-top: 0.85rem;
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
        .nav-divider {
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            margin: 0.1rem 0 0.85rem 0;
        }
        .brand-wrap {
            display: inline-flex;
            align-items: center;
            gap: 0.62rem;
        }
        .brand-mark {
            width: 30px;
            height: 30px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #e0f2fe;
            font-weight: 800;
            font-size: 0.78rem;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.9), rgba(34, 197, 94, 0.62));
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
        }
        .brand-name {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 750;
            letter-spacing: 0;
        }
        .nav-tools {
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .tool-btn {
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.52);
            border-radius: 999px;
            padding: 0.36rem 0.68rem;
            font-size: 0.78rem;
            line-height: 1;
        }
        .tool-btn:hover {
            border-color: rgba(56, 189, 248, 0.36);
            color: #f8fafc;
            background: rgba(30, 41, 59, 0.72);
        }
        .hero-shell {
            width: 100%;
            box-sizing: border-box;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background:
                radial-gradient(circle at 8% 0%, rgba(56, 189, 248, 0.12), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.92));
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.22);
            padding: 1.28rem 1.45rem 1.18rem 1.45rem;
            margin: 0 0 0.9rem 0;
        }
        .hero-content {
            max-width: 860px;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(56, 189, 248, 0.24);
            background: rgba(56, 189, 248, 0.075);
            color: #7dd3fc;
            border-radius: 999px;
            padding: 0.28rem 0.56rem;
            font-size: 0.66rem;
            font-weight: 750;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 2.18rem;
            line-height: 1.08;
            font-weight: 820;
            letter-spacing: -0.01em;
            margin-bottom: 0.45rem;
        }
        .hero-copy {
            color: #b6c2d2;
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 820px;
            margin-bottom: 0.8rem;
        }
        .hero-chips {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .hero-chip {
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.075);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            padding: 0.33rem 0.62rem;
            font-size: 0.73rem;
        }
        .top-controls-label {
            color: #94a3b8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.45rem;
            font-weight: 700;
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
            .top-nav {
                align-items: flex-start;
                gap: 0.7rem;
            flex-direction: column;
            }
            .dashboard-title {
                font-size: 1.55rem;
            }
            .hero-title {
                font-size: 1.72rem;
            }
            .hero-shell {
                padding: 1rem;
            }
        }
        .app-brand {
            display: inline-flex;
            align-items: center;
            gap: 0.62rem;
            min-height: 2.25rem;
        }
        .app-brand-mark {
            width: 30px;
            height: 30px;
            border-radius: 9px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #e0f2fe;
            font-weight: 800;
            font-size: 0.78rem;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.9), rgba(34, 197, 94, 0.62));
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
        }
        .app-brand-name {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 750;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.52);
            color: #cbd5e1;
            padding: 0.36rem 0.68rem;
            min-height: 2.05rem;
            font-size: 0.78rem;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            border-color: rgba(56, 189, 248, 0.36);
            color: #f8fafc;
            background: rgba(30, 41, 59, 0.72);
        }
        div[data-testid="stDownloadButton"] button:disabled,
        div[data-testid="stButton"] button:disabled {
            opacity: 0.52;
            cursor: not-allowed;
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


def _rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _render_top_nav(
    nav_slot,
    export_data: pd.DataFrame | None = None,
    export_filename: str = "stock_dashboard_export.csv",
) -> None:
    export_csv = export_data.to_csv(index=False).encode("utf-8") if export_data is not None else b""
    key_suffix = "ready" if export_data is not None else "pending"

    with nav_slot.container():
        nav_cols = st.columns([4.9, 0.92, 1.22, 0.82, 0.82])
        with nav_cols[0]:
            st.markdown(
                """
                <div class="brand-wrap">
                    <div class="brand-mark">MP</div>
                    <div class="brand-name">MarketPulse</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with nav_cols[1]:
            if st.button(
                "Compare",
                key=f"nav_compare_{key_suffix}",
                use_container_width=True,
                help="Jump to the Actual vs Predicted comparison chart.",
            ):
                st.session_state["jump_to_prediction_comparison"] = True
        with nav_cols[2]:
            if st.button(
                "Decision Summary",
                key=f"nav_decision_summary_{key_suffix}",
                use_container_width=True,
                help="Jump to the Decision Support Summary section.",
            ):
                st.session_state["jump_to_decision_summary"] = True
        with nav_cols[3]:
            st.download_button(
                "Export",
                data=export_csv,
                file_name=export_filename,
                mime="text/csv",
                key=f"nav_export_{key_suffix}",
                use_container_width=True,
                disabled=export_data is None,
                help="Download the current processed dataset with available technical indicators.",
            )
        with nav_cols[4]:
            if st.button(
                "Refresh",
                key=f"nav_refresh_{key_suffix}",
                use_container_width=True,
                help="Rerun the current dashboard analysis.",
            ):
                _load_prepared_data.clear()
                _run_models.clear()
                _search_symbols_cached.clear()
                _rerun_app()
        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)


def _jump_to_prediction_comparison() -> None:
    components.html(
        """
        <script>
        const targetId = "prediction-comparison-section";
        const openComparisonTab = () => {
            const doc = window.parent.document;
            const target = doc.getElementById(targetId);
            const tabs = Array.from(doc.querySelectorAll('[role="tab"]'));
            const comparisonTab = tabs.find((tab) => tab.innerText.trim().includes("Actual vs Predicted"));

            if (comparisonTab) {
                comparisonTab.click();
            }
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        };

        openComparisonTab();
        </script>
        """,
        height=0,
    )


def _jump_to_decision_summary() -> None:
    components.html(
        """
        <script>
        const scrollToDecisionSummary = () => {
            const target = window.parent.document.getElementById("decision-summary-section");
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        };

        scrollToDecisionSummary();
        </script>
        """,
        height=0,
    )


def _build_export_data(indicator_data: pd.DataFrame, results: dict, forecasts: dict) -> pd.DataFrame:
    metric_columns = {}
    for model_name in ("Baseline", "LSTM"):
        result = results.get(model_name)
        prefix = model_name.replace(" ", "_")
        metric_columns[f"{prefix}_MAE"] = result["metrics"]["mae"] if result else pd.NA
        metric_columns[f"{prefix}_RMSE"] = result["metrics"]["rmse"] if result else pd.NA
        metric_columns[f"{prefix}_MAPE"] = result["metrics"]["mape"] if result else pd.NA

    historical = indicator_data.copy()
    historical["Date"] = pd.to_datetime(historical["Date"])
    historical["type"] = "historical"
    historical["Actual_Close"] = historical["Close"]
    historical["Baseline_Prediction"] = pd.NA
    historical["LSTM_Prediction"] = pd.NA
    historical["Baseline_Forecast"] = pd.NA
    historical["LSTM_Forecast"] = pd.NA

    prediction_frames = []
    for model_name in ("Baseline", "LSTM"):
        result = results.get(model_name)
        if not result:
            continue

        actual = result["actual"][["Date", "Close"]].copy()
        predicted = result["predicted"][["Date", "Predicted_Close"]].copy()
        actual["Date"] = pd.to_datetime(actual["Date"])
        predicted["Date"] = pd.to_datetime(predicted["Date"])
        aligned = actual.merge(predicted, on="Date", how="inner").rename(columns={"Close": "Actual_Close"})
        aligned[f"{model_name}_Prediction"] = aligned["Predicted_Close"]
        prediction_frames.append(aligned[["Date", "Actual_Close", f"{model_name}_Prediction"]])

    if prediction_frames:
        predictions = pd.DataFrame({"Date": sorted(set().union(*(set(frame["Date"]) for frame in prediction_frames)))})
        for frame in prediction_frames:
            predictions = predictions.merge(frame, on="Date", how="left", suffixes=("", "_next"))
            if "Actual_Close_next" in predictions.columns:
                predictions["Actual_Close"] = predictions["Actual_Close"].combine_first(predictions["Actual_Close_next"])
                predictions = predictions.drop(columns=["Actual_Close_next"])
        predictions = predictions.sort_values("Date")
        predictions["type"] = "prediction"
        predictions["Close"] = predictions["Actual_Close"]
        predictions["Baseline_Forecast"] = pd.NA
        predictions["LSTM_Forecast"] = pd.NA
    else:
        predictions = pd.DataFrame()

    forecast_frames = []
    for model_name in ("Baseline", "LSTM"):
        forecast = forecasts.get(model_name)
        if forecast is None or forecast.empty:
            continue
        frame = forecast[["Date", "Predicted_Close"]].copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame[f"{model_name}_Forecast"] = frame["Predicted_Close"]
        forecast_frames.append(frame[["Date", f"{model_name}_Forecast"]])

    if forecast_frames:
        future_forecasts = forecast_frames[0]
        for frame in forecast_frames[1:]:
            future_forecasts = future_forecasts.merge(frame, on="Date", how="outer")
        future_forecasts = future_forecasts.sort_values("Date")
        future_forecasts["type"] = "forecast"
        future_forecasts["Actual_Close"] = pd.NA
        future_forecasts["Baseline_Prediction"] = pd.NA
        future_forecasts["LSTM_Prediction"] = pd.NA
    else:
        future_forecasts = pd.DataFrame()

    export_frames = []
    for frame in (historical, predictions, future_forecasts):
        if frame.empty:
            continue
        cleaned_frame = frame.dropna(axis=1, how="all")
        if not cleaned_frame.empty:
            export_frames.append(cleaned_frame)
    export_data = pd.concat(export_frames, ignore_index=True, sort=False)
    for column, value in metric_columns.items():
        export_data[column] = value

    ordered_columns = [
        "Date",
        "type",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA",
        "EMA",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Actual_Close",
        "Baseline_Prediction",
        "LSTM_Prediction",
        "Baseline_Forecast",
        "LSTM_Forecast",
        "Baseline_MAE",
        "Baseline_RMSE",
        "Baseline_MAPE",
        "LSTM_MAE",
        "LSTM_RMSE",
        "LSTM_MAPE",
    ]
    for column in ordered_columns:
        if column not in export_data.columns:
            export_data[column] = pd.NA

    export_data = export_data[ordered_columns].sort_values(["Date", "type"]).reset_index(drop=True)
    export_data["Date"] = pd.to_datetime(export_data["Date"]).dt.strftime("%Y-%m-%d")
    return export_data


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
        if not FINNHUB_API_KEY:
            st.info("Autocomplete unavailable. Enter ticker manually (e.g., AAPL, TSLA)")
            return normalize_ticker(cleaned_query or default_ticker)

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


def _render_prediction_explainer() -> None:
    st.info(
        "Baseline is a simple reference model using recent price behavior. "
        "LSTM is a pattern-based neural model that learns from historical sequences. "
        "MAE and RMSE measure average prediction error, with lower values indicating better fit. "
        "Forecasts are indicative decision support, not guaranteed outcomes."
    )


def _render_forecast_disclaimer() -> None:
    st.warning(
        "Forecasts shown here are model-based estimates derived from historical market data and technical indicators. "
        "They are intended for analytical and educational use only, and should not be interpreted as guaranteed outcomes "
        "or financial advice."
    )


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

    nav_slot = st.empty()
    _render_top_nav(nav_slot)

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-content">
                <div class="hero-eyebrow">MARKET INTELLIGENCE DASHBOARD</div>
                <div class="hero-title">Stock Market Trend Analysis</div>
                <div class="hero-copy">
                    Analyze price action, technical indicators, and short-term forecasts in a clean, chart-first analytics workspace.
                </div>
                <div class="hero-chips">
                    <span class="hero-chip">Historical Analysis</span>
                    <span class="hero-chip">Technical Indicators</span>
                    <span class="hero-chip">Forecast Insights</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="top-controls-label">Analysis Controls</div>', unsafe_allow_html=True)
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

    export_filename = f"{ticker}_{start_date.isoformat()}_{end_date.isoformat()}_analysis.csv"
    export_data = _build_export_data(indicator_data, results, forecasts)
    _render_top_nav(nav_slot, export_data, export_filename)

    for warning in model_warnings:
        st.warning(warning)

    valid_results = {name: result for name, result in results.items() if result is not None}
    preferred_model = compare_models(valid_results)["best_model"]
    summary = build_decision_summary(indicator_data, valid_results, forecasts, preferred_model)

    trend = classify_price_trend(indicator_data)
    _render_overview_cards(ticker, indicator_data, trend, preferred_model, forecast_horizon)

    st.markdown('<div class="section-title">Main Historical Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(create_close_price_chart(indicator_data, ticker), use_container_width=True, config=PLOTLY_CHART_CONFIG)

    st.markdown('<div class="section-title">Additional Market Data</div>', unsafe_allow_html=True)
    market_tabs = st.tabs(["Candlestick", "Volume"])
    with market_tabs[0]:
        st.plotly_chart(create_candlestick_chart(indicator_data, ticker), use_container_width=True, config=PLOTLY_CHART_CONFIG)
    with market_tabs[1]:
        st.plotly_chart(create_volume_chart(indicator_data, ticker), use_container_width=True, config=PLOTLY_CHART_CONFIG)

    st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)
    st.plotly_chart(create_indicator_panel(indicator_data, ticker), use_container_width=True, config=PLOTLY_CHART_CONFIG)

    st.markdown('<div id="prediction-comparison-section"></div><div class="section-title">Prediction</div>', unsafe_allow_html=True)
    if valid_results:
        prediction_tabs = st.tabs(["Model Comparison", "Actual vs Predicted", "Future Forecast"])
        with prediction_tabs[0]:
            _render_model_comparison(results, summary)
        with prediction_tabs[1]:
            st.plotly_chart(create_prediction_comparison_chart(valid_results), use_container_width=True, config=PLOTLY_CHART_CONFIG)
        with prediction_tabs[2]:
            st.plotly_chart(create_forecast_chart(indicator_data, forecasts, ticker, preferred_model), use_container_width=True, config=PLOTLY_CHART_CONFIG)
            _render_prediction_points(summary)
        _render_prediction_explainer()
        if st.session_state.pop("jump_to_prediction_comparison", False):
            _jump_to_prediction_comparison()
    else:
        st.warning("Prediction results are unavailable. Historical charts and technical indicators remain available.")

    _render_forecast_disclaimer()

    st.markdown('<div id="decision-summary-section"></div><div class="section-title">Decision-Support Summary</div>', unsafe_allow_html=True)
    _render_decision_summary(summary)
    if st.session_state.pop("jump_to_decision_summary", False):
        _jump_to_decision_summary()


if __name__ == "__main__":
    main()
