# Stock Market Trend Analysis Dashboard

A clean Streamlit dashboard for historical stock market visualization, technical indicators, model-based forecasting, and simple decision support.

## Scope

The project focuses only on:

- Historical OHLCV data visualization
- Technical indicators: SMA, EMA, RSI, MACD
- Predictive modeling with a baseline Linear Regression model and Keras LSTM
- Forecast visualization
- Explainable decision-support summary

It does not include authentication, portfolio tracking, alerts, news, or sentiment analysis.

## Project Structure

```text
stock_dashboard/
|-- app.py
|-- requirements.txt
|-- data/
|   `-- cache/
|-- utils/
|   |-- fetch_data.py
|   |-- preprocess.py
|   |-- indicators.py
|   `-- evaluation.py
|-- models/
|   |-- baseline_model.py
|   `-- lstm_model.py
|-- visualization/
|   `-- charts.py
|-- config/
|   `-- settings.py
`-- README.md
```

## Python Version

Use Python 3.11 or Python 3.12.

Do not use Python 3.14 for this project. The pinned scientific Python and TensorFlow stack may not have Python 3.14 wheels, so `pip` tries to build packages such as `pandas` from source and can fail with Meson or Visual Studio errors.

Check your active version:

```bash
python --version
```

## Setup On Windows PowerShell

From the `stock_dashboard` folder, remove the old virtual environment if it was created with Python 3.14:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
```

Install Python 3.11 from python.org or with winget:

```powershell
winget install Python.Python.3.11
```

Create a fresh virtual environment with Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Run the dashboard:

```powershell
python -m streamlit run app.py
```

## Setup On macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m streamlit run app.py
```

## Usage

1. Enter a stock ticker such as `AAPL` or `TSLA`.
2. Select a historical date range.
3. Choose `Baseline`, `LSTM`, or `Both`.
4. Select a forecast horizon.
5. Run the analysis.

The dashboard fetches data from yfinance, caches CSV files in `data/cache`, computes indicators manually with pandas, trains selected models, evaluates MAE/RMSE/MAPE, and displays forecasts with a decision-support summary.

## Troubleshooting

If you see an error like this:

```text
pandas ... Preparing metadata ... error
ERROR: Could not parse vswhere.exe output
```

Your virtual environment is probably using Python 3.14. Recreate `.venv` with Python 3.11 using the Windows PowerShell steps above.

If Yahoo Finance returns a message like this:

```text
YFTzMissingError: possibly delisted; No timezone found
```

Use the exchange ticker symbol instead of the company name. For example:

- `AAPL` for Apple
- `TSLA` for Tesla
- `NVDA` for NVIDIA
- `MSFT` for Microsoft

The app also includes a small alias map for common company names, but ticker symbols are more reliable.

If the LSTM model fails on Windows with a silent TensorFlow worker exit, restart the app after pulling the latest code. The LSTM implementation uses a conservative Keras `RNN(LSTMCell)` path and runs in a separate worker process so the Streamlit dashboard stays open even if TensorFlow fails.

## Notes

- LSTM training can take longer than the baseline model, especially on large date ranges.
- Forecasts are educational decision support and are not financial advice.
