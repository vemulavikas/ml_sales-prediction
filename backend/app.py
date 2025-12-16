from flask import Flask, request, jsonify
from flask_cors import CORS
from db import get_connection
from predict_lstm import run_prediction

app = Flask(__name__)
CORS(app)

MODEL_DIR = "../model_dir"
DATASET_PATH = "../Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv"


# ---------- ACTUAL SALES (FROM POSTGRESQL) ----------
@app.route("/actual", methods=["GET"])
def actual_sales():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT month, year, amount FROM actual_sales ORDER BY month")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    data = [
        {"month": r[0], "year": r[1], "amount": r[2], "type": "Actual"}
        for r in rows
    ]
    return jsonify({"data": data})


# ---------- THIS YEAR ANALYSIS (FROM POSTGRESQL) ----------
@app.route("/analysis", methods=["GET"])
def analysis():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT amount FROM actual_sales")
    amounts = [r[0] for r in cur.fetchall()]

    cur.close()
    conn.close()

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

    result = run_prediction(
        model_dir=MODEL_DIR,
        history_path=DATASET_PATH,
        predict_months=months
    )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM predicted_sales WHERE forecast_type = %s",
        (forecast_type,)
    )

    for p in result["predictions"]:
        cur.execute(
            "INSERT INTO predicted_sales (forecast_type, forecast_date, amount) VALUES (%s,%s,%s)",
            (forecast_type, p["date"], int(p["forecast"]))
        )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
