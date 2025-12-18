# ML Sales Forecast Web App

End-to-end sales forecasting app with a Flask backend + React frontend.

Key point: production `/forecast` is served from PostgreSQL (AWS RDS) for stability (no TensorFlow inference at request time).

## Project Structure

- `backend/`: Flask API, Postgres access, forecast loader scripts
- `frontend/`: React dashboard
- `models/`: trained model artifacts (SavedModel + metadata + scaler)
- `ml_training/`: training + ML utilities used for the assignment

## Backend API

- `GET /health`
- `GET /actual?year=2025` (monthly totals derived from the provided weekly-wide dataset)
- `GET /analysis?year=2025`
- `GET /forecast?months=12` or `months=24`

## Forecast Sources (Important)

The backend uses `FORECAST_MODE`:

- `db` (default, recommended for Render): reads from Postgres table `predicted_sales`
- `live`: runs TensorFlow inference on-demand (heavier; can be unstable on small instances)
- `precomputed`: legacy mode; if precomputed JSON files are missing it automatically falls back to DB

## Prerequisites

- Python 3.11
- Node.js 18+
- PostgreSQL (AWS RDS recommended)

## Environment Configuration

Copy the sample and fill in your values:

```bash
cp .env.example .env
```

Backend accepts either `POSTGRES_*` or `DB_*` environment variables.

## Local Setup

### Backend

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/app.py
```

Backend runs at http://127.0.0.1:5000.

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at http://localhost:3000.

## Load Data Into AWS RDS (one-time / whenever you reset)

These scripts create tables (if missing), delete existing rows, then reload:

- `backend/load_actual_to_db.py` loads monthly actuals into `actual_sales`
- `backend/load_predictions_to_db.py` generates predictions locally and inserts into `predicted_sales`
- `backend/reset_and_load_rds.py` runs both in a single step

Run (after setting your `.env` with RDS credentials):

```bash
python backend/reset_and_load_rds.py
```

## Deploy Backend To Render (final)

This repo includes `render.yaml` configured for a Python web service.

1. Create a new Render **Web Service** from your GitHub repo.
2. Render will read `render.yaml`.
3. In Render → Environment, set your DB credentials (either `DB_*` or `POSTGRES_*`).
4. Ensure `FORECAST_MODE=db` (recommended).

After deploy:

- `https://<your-render-service>.onrender.com/health`
- `https://<your-render-service>.onrender.com/forecast?months=12`

## ML Training (Assignment)

The ML training scripts live in `ml_training/`.

### Model Metrics

From `models/metadata.json` (latest exported training run):

- Validation MAE: 50.53
- Validation RMSE: 58.22
- Validation MAPE: 0.72%

Train:

```bash
py -3.11 ml_training/train_lstm.py \
  --input Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv \
  --model_out models \
  --epochs 50
```

Generate a forecast JSON (ML-only):

```bash
py -3.11 ml_training/predict_lstm.py \
  --model_dir models \
  --history Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv \
  --predict_months 12 \
  --out test_forecast.json
```
