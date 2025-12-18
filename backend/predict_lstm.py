import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf


def _load_saved_model_for_inference(saved_model_dir: str):
	"""Load a TensorFlow SavedModel for inference with a .predict(x) API.

	We avoid keras.layers.TFSMLayer here because it can be environment-sensitive
	(and has caused crashes in some deploy targets). This loader prefers
	tf.keras.models.load_model when available, and falls back to calling a
	SavedModel signature directly.
	"""
	# Preferred: tf.keras model load
	try:
		model = tf.keras.models.load_model(saved_model_dir)
		# Sanity: ensure it has predict
		_ = getattr(model, "predict")
		return model
	except Exception:
		pass

	# Fallback: signature-based inference
	loaded = tf.saved_model.load(saved_model_dir)
	infer = loaded.signatures.get("serving_default")
	if infer is None:
		infer = next(iter(loaded.signatures.values()))

	class _SignatureModel:
		def __init__(self, infer_fn):
			self._infer = infer_fn

		def predict(self, x, verbose=0):  # noqa: ARG002 - verbose kept for API compat
			x_tf = tf.convert_to_tensor(x)
			try:
				_, kw = self._infer.structured_input_signature
				expected = list(kw.keys())
			except Exception:
				expected = []

			if expected:
				outputs = self._infer(**{expected[0]: x_tf})
			else:
				outputs = self._infer(x_tf)

			if isinstance(outputs, dict):
				outputs = next(iter(outputs.values()))
			return outputs.numpy()

	return _SignatureModel(infer)


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


def _fit_minmax_scaler(values_train: np.ndarray, feature_range=(0.0, 1.0)) -> dict:
	"""Fit MinMax scaling params (sklearn-compatible) without sklearn.

	We store parameters in a dict so we never need to unpickle a scaler.
	This avoids deployment failures caused by scikit-learn version mismatch.
	"""
	if values_train.ndim != 2:
		raise ValueError("values_train must have shape (n_samples, n_features)")
	min_range, max_range = map(float, feature_range)
	data_min = np.nanmin(values_train, axis=0).astype("float32")
	data_max = np.nanmax(values_train, axis=0).astype("float32")
	data_range = data_max - data_min

	# Avoid divide-by-zero if a feature is constant
	safe_range = np.where(data_range == 0.0, 1.0, data_range).astype("float32")
	scale = ((max_range - min_range) / safe_range).astype("float32")
	min_ = (min_range - data_min * scale).astype("float32")

	return {
		"feature_range": (min_range, max_range),
		"data_min": data_min,
		"data_max": data_max,
		"scale": scale,
		"min": min_,
		"n_features": int(values_train.shape[1]),
	}


def _minmax_transform(values: np.ndarray, scaler_params: dict) -> np.ndarray:
	if values.ndim != 2:
		raise ValueError("values must have shape (n_samples, n_features)")
	scale = np.asarray(scaler_params["scale"], dtype="float32")
	min_ = np.asarray(scaler_params["min"], dtype="float32")
	return (values.astype("float32") * scale) + min_


def _minmax_inverse_first_feature(scaled_first: np.ndarray, scaler_params: dict) -> np.ndarray:
	"""Inverse-transform the first feature only.

	MinMaxScaler transforms each feature independently, so we can invert the
	first feature directly without needing dummy values for other features.
	"""
	if scaled_first.ndim == 1:
		scaled_first = scaled_first.reshape(-1, 1)

	scale0 = float(np.asarray(scaler_params["scale"], dtype="float32")[0])
	min0 = float(np.asarray(scaler_params["min"], dtype="float32")[0])

	# If a feature was constant, we set safe_range=1.0, so scale0 is finite.
	inv = (scaled_first.astype("float32") - min0) / (scale0 if scale0 != 0.0 else 1.0)
	return inv.reshape(-1)


def iterative_forecast(
	model,
	scaler_params: dict,
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
	scaled_history = _minmax_transform(values, scaler_params).astype("float32")
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
	forecasts = _minmax_inverse_first_feature(scaled_arr, scaler_params)

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

	model = _load_saved_model_for_inference(saved_model_dir)

	history_df = load_sales_file(history_path)

	# Fit scaling params exactly like training: fit on training portion only
	# and reserve the last 12 steps for validation.
	values = history_df.values.astype("float32")
	n_total = values.shape[0]
	n_val_steps = 12
	if n_total < lookback + n_val_steps:
		raise ValueError(
			"Time series is too short. Need at least lookback + 12 steps of data."
		)
	n_train = n_total - n_val_steps
	scaler_params = _fit_minmax_scaler(values[:n_train])

	future_dates_raw, forecasts = iterative_forecast(
		model=model,
		scaler_params=scaler_params,
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
