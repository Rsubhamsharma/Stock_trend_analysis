import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _base_layout(title: str, yaxis_title: str) -> dict:
    return {
        "title": title,
        "xaxis_title": "Date",
        "yaxis_title": yaxis_title,
        "template": "plotly_white",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.02, "x": 0},
        "margin": {"l": 30, "r": 30, "t": 70, "b": 30},
    }


def create_close_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close", line={"color": "#2563eb"}))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], mode="lines", name="SMA", line={"color": "#f59e0b"}))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA"], mode="lines", name="EMA", line={"color": "#10b981"}))
    fig.update_layout(**_base_layout(f"{ticker} Closing Price with SMA and EMA", "Price"))
    return fig


def create_candlestick_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
        )
    )
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], mode="lines", name="SMA", line={"color": "#f59e0b"}))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA"], mode="lines", name="EMA", line={"color": "#10b981"}))
    fig.update_layout(**_base_layout(f"{ticker} OHLC Candlestick Chart", "Price"))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def create_volume_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], name="Volume", marker_color="#64748b"))
    fig.update_layout(**_base_layout(f"{ticker} Trading Volume", "Volume"))
    return fig


def create_prediction_comparison_chart(actual: pd.DataFrame, predicted: pd.DataFrame, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["Date"], y=actual["Close"], mode="lines", name="Actual", line={"color": "#111827"}))
    fig.add_trace(
        go.Scatter(
            x=predicted["Date"],
            y=predicted["Predicted_Close"],
            mode="lines",
            name=f"{model_name} Prediction",
            line={"color": "#dc2626"},
        )
    )
    fig.update_layout(**_base_layout(f"{model_name}: Predicted vs Actual Close", "Price"))
    return fig


def create_forecast_chart(data: pd.DataFrame, forecasts: dict, ticker: str) -> go.Figure:
    fig = go.Figure()
    historical_tail = data.tail(120)
    fig.add_trace(
        go.Scatter(
            x=historical_tail["Date"],
            y=historical_tail["Close"],
            mode="lines",
            name="Historical Close",
            line={"color": "#111827"},
        )
    )

    colors = {"Baseline": "#f97316", "LSTM": "#2563eb"}
    for model_name, forecast in forecasts.items():
        if forecast is None or forecast.empty:
            continue
        joined = pd.concat(
            [
                pd.DataFrame({"Date": [data["Date"].iloc[-1]], "Predicted_Close": [data["Close"].iloc[-1]]}),
                forecast,
            ],
            ignore_index=True,
        )
        fig.add_trace(
            go.Scatter(
                x=joined["Date"],
                y=joined["Predicted_Close"],
                mode="lines+markers",
                name=f"{model_name} Forecast",
                line={"color": colors.get(model_name, "#7c3aed")},
            )
        )

    fig.update_layout(**_base_layout(f"{ticker} Future Forecast", "Price"))
    return fig


def create_indicator_panel(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"{ticker} RSI", "MACD"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", name="RSI", line={"color": "#7c3aed"}), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=1, col=1)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], mode="lines", name="MACD", line={"color": "#2563eb"}), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=data["Date"], y=data["MACD_Signal"], mode="lines", name="Signal", line={"color": "#f59e0b"}),
        row=2,
        col=1,
    )
    fig.add_trace(go.Bar(x=data["Date"], y=data["MACD_Histogram"], name="Histogram", marker_color="#94a3b8"), row=2, col=1)
    fig.update_layout(template="plotly_white", hovermode="x unified", height=650, margin={"l": 30, "r": 30, "t": 70, "b": 30})
    return fig
