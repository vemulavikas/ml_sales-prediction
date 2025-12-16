import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def _load_standard_date_sales(df: pd.DataFrame) -> pd.DataFrame:
	"""Load standard date/sales data and aggregate to monthly totals."""
	if "date" not in df.columns or "sales" not in df.columns:
		raise ValueError("Input file must contain 'date' and 'sales' columns")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.dropna(subset=["date"])
	if df.empty:
		raise ValueError("No valid dates found in input file")

	df = df.sort_values("date").set_index("date")
	monthly = df["sales"].resample("MS").sum()

	monthly = monthly.astype(float)
	monthly = monthly.interpolate(method="linear").ffill().bfill()

	if monthly.isna().any():
		raise ValueError("Failed to clean series; NaNs remain after filling.")

	ts_df = monthly.to_frame(name="sales")
	ts_df.attrs["date_mode"] = "date_sales_monthly"
	return ts_df


def _load_weekly_wide_sales(df: pd.DataFrame) -> pd.DataFrame:
	"""Mirror of train_lstm._load_weekly_wide_sales for prediction.

	Produces the same two-feature time series used during training:
	- 'sales': total weekly sales across all products
	- 'norm': (optional) mean of "Normalized i" across all products
	"""
	weekly_cols = [
		c
		for c in df.columns
		if isinstance(c, str)
		and c.strip().upper().startswith("W")
		and c.strip()[1:].isdigit()
	]

	if len(weekly_cols) < 4:
		raise ValueError(
			"Could not detect weekly W0..Wn columns; file is not in the expected "
			"weekly wide format and also lacks 'date'/'sales'."
		)

	weekly_cols_sorted = sorted(weekly_cols, key=lambda c: int(c.strip()[1:]))
	weekly_sum = df[weekly_cols_sorted].sum(axis=0).astype(float)

	weekly_index = pd.date_range(
		start="2000-01-03", periods=len(weekly_cols_sorted), freq="W-MON"
	)

	sales_series = pd.Series(weekly_sum.values, index=weekly_index, name="sales")

	norm_cols = [
		c
		for c in df.columns
		if isinstance(c, str)
		and c.strip().startswith("Normalized")
		and c.strip().split(" ")[-1].isdigit()
	]

	if norm_cols:
		norm_cols_sorted = sorted(
			norm_cols, key=lambda c: int(c.strip().split(" ")[-1])
		)
		if len(norm_cols_sorted) >= len(weekly_cols_sorted):
			norm_cols_sorted = norm_cols_sorted[: len(weekly_cols_sorted)]
		norm_mean = df[norm_cols_sorted].mean(axis=0).astype(float)
		norm_series = pd.Series(norm_mean.values, index=weekly_index, name="norm")
		ts_df = pd.concat([sales_series, norm_series], axis=1)
	else:
		ts_df = sales_series.to_frame()

	ts_df = ts_df.astype(float)
	ts_df = ts_df.interpolate(method="linear").ffill().bfill()

	if ts_df.isna().any().any():
		raise ValueError("Failed to clean weekly series derived from weekly data.")

	ts_df.attrs["date_mode"] = "synthetic_weekly"
	return ts_df


def load_sales_file(path: str) -> pd.DataFrame:
	"""Load a sales time series from CSV or Excel for prediction."""
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Input file not found: {path}")

	ext = os.path.splitext(path)[1].lower()
	try:
		if ext == ".csv":
			df = pd.read_csv(path)
		elif ext in {".xlsx", ".xls"}:
			df = pd.read_excel(path)
		else:
			raise ValueError("Unsupported file extension. Use .csv or .xlsx")
	except Exception as exc:  # pragma: no cover - I/O related
		raise RuntimeError(f"Failed to read input file: {exc}") from exc

	if "date" in df.columns and "sales" in df.columns:
		return _load_standard_date_sales(df)

	return _load_weekly_wide_sales(df)


def _inverse_first_feature(scaled_col: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
	"""Inverse-transform the first feature using a multi-feature scaler."""
	if scaled_col.ndim == 1:
		scaled_col = scaled_col.reshape(-1, 1)

	n_samples = scaled_col.shape[0]
	n_features = getattr(scaler, "n_features_in_", 1)

	if n_features == 1:
		return scaler.inverse_transform(scaled_col).reshape(-1)

	zeros = np.zeros((n_samples, n_features - 1), dtype="float32")
	full_scaled = np.concatenate([scaled_col.astype("float32"), zeros], axis=1)
	inv_full = scaler.inverse_transform(full_scaled)
	return inv_full[:, 0]


def iterative_forecast(
	model: tf.keras.Model,
	scaler: MinMaxScaler,
	history_df: pd.DataFrame,
	lookback: int,
	predict_months: int,
):
	"""Iteratively forecast `predict_months` steps ahead.

	Uses the same multi-feature structure as training:
	- The model outputs a scaled prediction for the first feature only
	  (total sales).
	- For each new step, we build a new feature vector by copying the
	  last known feature vector and replacing its first element with the
	  predicted scaled value.
	"""
	if predict_months <= 0:
		raise ValueError("predict_months must be positive")
	if len(history_df) < lookback:
		raise ValueError("Not enough history to run iterative forecast")

	values = history_df.values.astype("float32")
	scaled_history = scaler.transform(values).astype("float32")
	n_features = scaled_history.shape[1]

	history_list = scaled_history.tolist()
	scaled_predictions = []

	for _ in range(predict_months):
		window = np.array(history_list[-lookback:], dtype="float32").reshape(
			1, lookback, n_features
		)
		scaled_pred = model.predict(window, verbose=0)[0, 0]
		scaled_predictions.append(float(scaled_pred))

		last_vec = history_list[-1]
		new_vec = list(last_vec)
		new_vec[0] = float(scaled_pred)
		history_list.append(new_vec)

	scaled_arr = np.array(scaled_predictions, dtype="float32").reshape(-1, 1)
	forecasts = _inverse_first_feature(scaled_arr, scaler)

	# Base monthly index; may be remapped depending on date_mode
	last_date = history_df.index[-1]
	future_dates = pd.date_range(last_date, periods=predict_months + 1, freq="MS")[1:]
	return future_dates, forecasts


def _map_output_dates(
	future_dates: pd.DatetimeIndex,
	predict_months: int,
	metadata: dict,
	history_df: pd.DataFrame,
) -> pd.DatetimeIndex:
	"""Map raw future_dates to business-meaningful calendar months.

	For synthetic weekly data (assignment dataset), we treat the
	available history as the "current" year and label forecasts as the
	next calendar year (and beyond if horizon > 12 months).
	"""
	date_mode = metadata.get("date_mode") or history_df.attrs.get(
		"date_mode", "unknown"
	)

	if date_mode != "synthetic_weekly":
		return future_dates

	now_year = datetime.now().year
	start_year = now_year + 1

	months = []
	year = start_year
	month = 1
	for _ in range(predict_months):
		months.append(datetime(year, month, 1))
		month += 1
		if month > 12:
			month = 1
			year += 1

	return pd.DatetimeIndex(months)


def load_scaler(model_dir: str) -> MinMaxScaler:
	scaler_path = os.path.join(model_dir, "scaler.pkl")
	if not os.path.isfile(scaler_path):
		raise FileNotFoundError(f"Scaler not found at {scaler_path}")

	try:
		import joblib

		scaler = joblib.load(scaler_path)
	except Exception:  # pragma: no cover - serialization specifics
		import pickle

		with open(scaler_path, "rb") as f:
			scaler = pickle.load(f)
	return scaler


def load_metadata(model_dir: str) -> dict:
	metadata_path = os.path.join(model_dir, "metadata.json")
	if not os.path.isfile(metadata_path):
		raise FileNotFoundError(f"metadata.json not found in {model_dir}")
	with open(metadata_path, "r", encoding="utf-8") as f:
		return json.load(f)


def run_prediction(
	model_dir: str,
	history_path: str,
	predict_months: int,
	out_path: str | None = None,
) -> dict:
	"""Run forecast and optionally write JSON output to disk."""
	metadata = load_metadata(model_dir)
	lookback = int(metadata.get("lookback", 12))

	saved_model_dir = os.path.join(model_dir, "lstm_saved_model")
	if not os.path.isdir(saved_model_dir):
		raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

	model = tf.keras.models.load_model(saved_model_dir)
	scaler = load_scaler(model_dir)

	history_df = load_sales_file(history_path)

	future_dates_raw, forecasts = iterative_forecast(
		model=model,
		scaler=scaler,
		history_df=history_df,
		lookback=lookback,
		predict_months=predict_months,
	)

	mapped_dates = _map_output_dates(
		future_dates=future_dates_raw,
		predict_months=predict_months,
		metadata=metadata,
		history_df=history_df,
	)

	predictions = [
		{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
		for d, v in zip(mapped_dates, forecasts)
	]

	result = {
		"predictions": predictions,
		"generated_at": datetime.now(timezone.utc).isoformat(),
	}

	if out_path:
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(result, f, indent=2)

	return result


def forecast_to_dataframe(
	model_dir: str,
	history_path: str,
	predict_months: int,
) -> pd.DataFrame:
	"""Return a DataFrame with mapped forecast dates and values."""
	metadata = load_metadata(model_dir)
	lookback = int(metadata.get("lookback", 12))

	saved_model_dir = os.path.join(model_dir, "lstm_saved_model")
	if not os.path.isdir(saved_model_dir):
		raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

	model = tf.keras.models.load_model(saved_model_dir)
	scaler = load_scaler(model_dir)

	history_df = load_sales_file(history_path)

	future_dates_raw, forecasts = iterative_forecast(
		model=model,
		scaler=scaler,
		history_df=history_df,
		lookback=lookback,
		predict_months=predict_months,
	)

	mapped_dates = _map_output_dates(
		future_dates=future_dates_raw,
		predict_months=predict_months,
		metadata=metadata,
		history_df=history_df,
	)

	df = pd.DataFrame({"date": mapped_dates, "forecast": forecasts})
	df["date"] = pd.to_datetime(df["date"])  # ensure datetime dtype
	return df


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run LSTM-based sales forecast using a trained model."
	)
	parser.add_argument(
		"--model_dir",
		required=True,
		help="Directory containing the trained model, scaler, and metadata.json.",
	)
	parser.add_argument(
		"--history",
		required=True,
		help="Path to the historical sales file used for forecasting.",
	)
	parser.add_argument(
		"--predict_months",
		type=int,
		default=12,
		help="Number of future months to forecast (default: 12)",
	)
	parser.add_argument(
		"--out",
		default=None,
		help="Optional path to write forecast JSON output.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	try:
		result = run_prediction(
			model_dir=args.model_dir,
			history_path=args.history,
			predict_months=args.predict_months,
			out_path=args.out,
		)
	except Exception as exc:
		print(f"Error: {exc}")
		raise SystemExit(1)

	if not args.out:
		print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
