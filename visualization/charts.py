import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "background": "#0f172a",
    "paper": "#111827",
    "grid": "#263244",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "close": "#38bdf8",
    "sma": "#f59e0b",
    "ema": "#22c55e",
    "baseline": "#fb923c",
    "lstm": "#a78bfa",
    "actual": "#f8fafc",
    "danger": "#ef4444",
    "success": "#22c55e",
}

SPIKE_COLOR = "rgba(203, 213, 225, 0.42)"


def _price_hover(name: str) -> str:
    return f"%{{x|%b %d, %Y}}<br>{name}: $%{{y:,.2f}}<extra></extra>"


def _value_hover(name: str, suffix: str = "") -> str:
    return f"%{{x|%b %d, %Y}}<br>{name}: %{{y:,.2f}}{suffix}<extra></extra>"


def _base_layout(title: str, yaxis_title: str, height: int = 430) -> dict:
    return {
        "title": {"text": title, "font": {"size": 18}},
        "xaxis_title": None,
        "yaxis_title": yaxis_title,
        "template": "plotly_dark",
        "height": height,
        "hovermode": "x unified",
        "dragmode": "pan",
        "paper_bgcolor": COLORS["paper"],
        "plot_bgcolor": COLORS["background"],
        "font": {"color": COLORS["text"]},
        "legend": {"orientation": "h", "y": 1.08, "x": 0},
        "margin": {"l": 35, "r": 25, "t": 70, "b": 35},
    }


def _style_axes(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        hoverlabel={
            "bgcolor": COLORS["paper"],
            "bordercolor": COLORS["grid"],
            "font": {"color": COLORS["text"], "size": 12},
        }
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=SPIKE_COLOR,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        showspikes=True,
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=SPIKE_COLOR,
    )
    return fig


def create_close_price_chart(data: pd.DataFrame, ticker: str, height: int = 620) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close", line={"color": COLORS["close"], "width": 2.4}, hovertemplate=_price_hover("Close")))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], mode="lines", name="SMA", line={"color": COLORS["sma"], "width": 1.5}, hovertemplate=_price_hover("SMA")))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA"], mode="lines", name="EMA", line={"color": COLORS["ema"], "width": 1.5}, hovertemplate=_price_hover("EMA")))
    fig.update_layout(**_base_layout(f"{ticker} Price Action", "Price", height))
    fig.update_xaxes(rangeslider_visible=False)
    return _style_axes(fig)


def create_candlestick_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    hover_text = (
        pd.to_datetime(data["Date"]).dt.strftime("%b %d, %Y")
        + "<br>Open: $"
        + data["Open"].map("{:,.2f}".format)
        + "<br>High: $"
        + data["High"].map("{:,.2f}".format)
        + "<br>Low: $"
        + data["Low"].map("{:,.2f}".format)
        + "<br>Close: $"
        + data["Close"].map("{:,.2f}".format)
    )
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            text=hover_text,
            hoverinfo="text",
        )
    )
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], mode="lines", name="SMA", line={"color": COLORS["sma"], "width": 1.2}, hovertemplate=_price_hover("SMA")))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA"], mode="lines", name="EMA", line={"color": COLORS["ema"], "width": 1.2}, hovertemplate=_price_hover("EMA")))
    fig.update_layout(**_base_layout(f"{ticker} Candlestick", "Price", 480))
    fig.update_xaxes(rangeslider_visible=False)
    return _style_axes(fig)


def create_volume_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], name="Volume", marker_color="#475569", hovertemplate="%{x|%b %d, %Y}<br>Volume: %{y:,.0f}<extra></extra>"))
    fig.update_layout(**_base_layout(f"{ticker} Volume", "Volume", 360))
    return _style_axes(fig)


def create_prediction_comparison_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"Baseline": COLORS["baseline"], "LSTM": COLORS["lstm"]}
    aligned_frames = []

    for model_name, result in results.items():
        if result is None:
            continue

        actual = result["actual"][["Date", "Close"]].copy()
        predicted = result["predicted"][["Date", "Predicted_Close"]].copy()
        actual["Date"] = pd.to_datetime(actual["Date"])
        predicted["Date"] = pd.to_datetime(predicted["Date"])

        aligned = actual.merge(predicted, on="Date", how="inner").sort_values("Date")
        if aligned.empty:
            continue

        aligned_frames.append(aligned[["Date", "Close"]])
        fig.add_trace(
            go.Scatter(
                x=aligned["Date"],
                y=aligned["Predicted_Close"],
                mode="lines",
                name=f"{model_name} predicted",
                line={"color": colors.get(model_name, "#f472b6"), "width": 2},
                hovertemplate=_price_hover(f"{model_name} predicted"),
            )
        )

    if aligned_frames:
        actual_by_date = (
            pd.concat(aligned_frames, ignore_index=True)
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
        )
        fig.add_trace(
            go.Scatter(
                x=actual_by_date["Date"],
                y=actual_by_date["Close"],
                mode="lines",
                name="Actual",
                line={"color": COLORS["actual"], "width": 2.4},
                hovertemplate=_price_hover("Actual"),
            )
        )

    fig.update_layout(**_base_layout("Actual vs Predicted Close", "Price", 470))
    return _style_axes(fig)


def create_forecast_chart(data: pd.DataFrame, forecasts: dict, ticker: str, preferred_model: str | None = None) -> go.Figure:
    fig = go.Figure()
    historical_tail = data.tail(140)
    fig.add_trace(
        go.Scatter(
            x=historical_tail["Date"],
            y=historical_tail["Close"],
            mode="lines",
            name="Historical close",
            line={"color": COLORS["actual"], "width": 2},
            hovertemplate=_price_hover("Historical close"),
        )
    )

    if preferred_model in forecasts and forecasts.get(preferred_model) is not None:
        visible_forecasts = {preferred_model: forecasts[preferred_model]}
    else:
        visible_forecasts = forecasts

    colors = {"Baseline": COLORS["baseline"], "LSTM": COLORS["lstm"]}
    last_date = data["Date"].iloc[-1]
    for model_name, forecast in visible_forecasts.items():
        if forecast is None or forecast.empty:
            continue
        joined = pd.concat(
            [
                pd.DataFrame({"Date": [last_date], "Predicted_Close": [data["Close"].iloc[-1]]}),
                forecast,
            ],
            ignore_index=True,
        )
        fig.add_trace(
            go.Scatter(
                x=joined["Date"],
                y=joined["Predicted_Close"],
                mode="lines+markers",
                name=f"{model_name} forecast",
                line={"color": colors.get(model_name, "#f472b6"), "width": 2.4, "dash": "dot"},
                marker={"size": 6},
                hovertemplate=_price_hover(f"{model_name} forecast"),
            )
        )

    max_y = max(data["Close"].tail(140).max(), *(f["Predicted_Close"].max() for f in visible_forecasts.values() if f is not None and not f.empty))
    min_y = min(data["Close"].tail(140).min(), *(f["Predicted_Close"].min() for f in visible_forecasts.values() if f is not None and not f.empty))
    fig.add_shape(
        type="rect",
        x0=last_date,
        x1=max((forecast["Date"].max() for forecast in visible_forecasts.values() if forecast is not None and not forecast.empty), default=last_date),
        y0=min_y,
        y1=max_y,
        fillcolor="#1e293b",
        opacity=0.35,
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(**_base_layout(f"{ticker} Forecast", "Price", 470))
    return _style_axes(fig)


def create_indicator_panel(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=(f"{ticker} RSI", "MACD"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", name="RSI", line={"color": COLORS["lstm"], "width": 2}, hovertemplate=_value_hover("RSI")), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"], row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"], row=1, col=1)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], mode="lines", name="MACD", line={"color": COLORS["close"], "width": 2}, hovertemplate=_value_hover("MACD")), row=2, col=1)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], mode="lines", name="Signal", line={"color": COLORS["sma"], "width": 1.8}, hovertemplate=_value_hover("Signal")), row=2, col=1)
    fig.add_trace(go.Bar(x=data["Date"], y=data["MACD_Histogram"], name="Histogram", marker_color="#64748b", hovertemplate=_value_hover("Histogram")), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["background"],
        font={"color": COLORS["text"]},
        hovermode="x unified",
        dragmode="pan",
        height=620,
        legend={"orientation": "h", "y": 1.05, "x": 0},
        margin={"l": 35, "r": 25, "t": 70, "b": 35},
    )
    return _style_axes(fig)
