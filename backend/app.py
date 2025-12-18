import os

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover
    CORS = None

import pandas as pd

from predict_lstm import run_prediction

app = Flask(__name__)
if CORS is not None:
    CORS(app)

# Resolve paths relative to the project root so they work regardless of
# where the app is started from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(
    BASE_DIR, "Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv"
)


def _load_actual_monthly_from_dataset(year: int) -> list[dict]:
    """Return monthly totals derived from the assignment weekly-wide dataset.

    Output matches the frontend expectation:
    [{month: 'Jan', year: 2025, amount: 1234, type: 'Actual'}, ...]
    """
    df = pd.read_csv(DATASET_PATH)
    weekly_cols = [
        c
        for c in df.columns
        if isinstance(c, str)
        and c.strip().upper().startswith("W")
        and c.strip()[1:].isdigit()
    ]
    if not weekly_cols:
        raise ValueError("No weekly columns (W0..Wn) found in dataset")

    weekly_cols_sorted = sorted(weekly_cols, key=lambda c: int(c.strip()[1:]))
    weekly_sum = df[weekly_cols_sorted].sum(axis=0).astype(float).values

    # Map weekly totals onto real-ish dates for the requested calendar year.
    start = pd.Timestamp(year=year, month=1, day=1)
    start_monday = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
    weekly_dates = pd.date_range(
        start=start_monday, periods=len(weekly_sum), freq="W-MON"
    )

    weekly_series = pd.Series(weekly_sum, index=weekly_dates)
    monthly = weekly_series.resample("MS").sum()

    out = []
    for d, v in monthly.items():
        out.append(
            {
                "month": d.strftime("%b"),
                "year": int(d.year),
                "amount": float(v),
                "type": "Actual",
            }
        )
    return out


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "ok",
            "message": "ML Sales Prediction API is running",
            "endpoints": {
                "actual": "/actual",
                "analysis": "/analysis",
                "forecast": "/forecast?months=12",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    versions = {"python": None, "tensorflow": None, "numpy": None, "pandas": None}
    try:
        import sys

        versions["python"] = sys.version.split()[0]
    except Exception:
        pass
    try:
        import tensorflow as tf

        versions["tensorflow"] = getattr(tf, "__version__", None)
    except Exception:
        pass
    try:
        import numpy as np

        versions["numpy"] = getattr(np, "__version__", None)
    except Exception:
        pass
    try:
        versions["pandas"] = getattr(pd, "__version__", None)
    except Exception:
        pass

    return jsonify(
        {
            "status": "ok",
            "versions": versions,
            "model_dir_exists": os.path.isdir(MODEL_DIR),
            "dataset_exists": os.path.isfile(DATASET_PATH),
        }
    )


# ---------- ACTUAL SALES (FROM DATASET) ----------
@app.route("/actual", methods=["GET"])
def actual_sales():
    year = request.args.get("year", default=pd.Timestamp.utcnow().year, type=int)
    try:
        data = _load_actual_monthly_from_dataset(year=year)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"data": data})


# ---------- THIS YEAR ANALYSIS (FROM DATASET) ----------
@app.route("/analysis", methods=["GET"])
def analysis():
    year = request.args.get("year", default=pd.Timestamp.utcnow().year, type=int)
    try:
        rows = _load_actual_monthly_from_dataset(year=year)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    amounts = [float(r["amount"]) for r in rows]
    if not amounts:
        return jsonify({"error": "No amounts found"}), 400

    analysis = {
        "total_sales": sum(amounts),
        "average_sales": round(sum(amounts) / len(amounts), 2),
        "highest_month": amounts.index(max(amounts)) + 1,
        "lowest_month": amounts.index(min(amounts)) + 1,
    }
    return jsonify({"analysis": analysis})


# ---------- FORECAST (ML + SAVE TO POSTGRESQL) ----------
@app.route("/forecast", methods=["GET"])
def forecast():
    months = request.args.get("months", 12, type=int)
    forecast_type = "1year" if months == 12 else "2year"

    # ------------------------------------------------------------------
    # Render stability: TensorFlow inference can crash the worker on small
    # instances. If we're on Render (or explicitly configured), serve
    # precomputed forecasts generated from the same trained model.
    # ------------------------------------------------------------------
    forecast_mode = os.environ.get("FORECAST_MODE", "auto").lower()
    on_render = bool(
        os.environ.get("RENDER")
        or os.environ.get("RENDER_SERVICE_ID")
        or os.environ.get("RENDER_EXTERNAL_URL")
    )

    if forecast_mode == "precomputed" or (forecast_mode == "auto" and on_render):
        precomputed_path = os.path.join(
            os.path.dirname(__file__),
            "precomputed",
            f"forecast_{months}.json",
        )
        try:
            import json

            with open(precomputed_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["precomputed"] = True
        except Exception as exc:
            return jsonify({"error": f"Precomputed forecast unavailable: {exc}"}), 500
    else:
        # Local/dev: run live inference.
        result = run_prediction(
            model_dir=MODEL_DIR,
            history_path=DATASET_PATH,
            predict_months=months,
        )

    # Best-effort persistence (not required for assignment).
    try:
        from db import get_connection  # local import to avoid hard dependency

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            "DELETE FROM predicted_sales WHERE forecast_type = %s",
            (forecast_type,),
        )

        for p in result.get("predictions", []):
            cur.execute(
                "INSERT INTO predicted_sales (forecast_type, forecast_date, amount) VALUES (%s,%s,%s)",
                (forecast_type, p["date"], int(p["forecast"])),
            )

        conn.commit()
        cur.close()
        conn.close()
        result["db_saved"] = True
    except Exception:
        result["db_saved"] = False

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
