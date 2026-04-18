import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from models.lstm_model import evaluate_and_forecast_lstm_local


def _frame_to_records(frame: pd.DataFrame) -> list[dict]:
    serializable = frame.copy()
    serializable["Date"] = pd.to_datetime(serializable["Date"]).dt.strftime("%Y-%m-%d")
    return serializable.to_dict(orient="records")


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python -m models.lstm_worker <input_csv> <output_json> <horizon>", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    horizon = int(sys.argv[3])

    data = pd.read_csv(input_path, parse_dates=["Date"])
    evaluation, forecast = evaluate_and_forecast_lstm_local(data, horizon)

    payload = {
        "evaluation": {
            "metrics": evaluation["metrics"],
            "actual": _frame_to_records(evaluation["actual"]),
            "predicted": _frame_to_records(evaluation["predicted"]),
        },
        "forecast": _frame_to_records(forecast),
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
