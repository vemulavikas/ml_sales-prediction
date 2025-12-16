import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


def _load_standard_date_sales(df: pd.DataFrame) -> pd.DataFrame:
	"""Load standard date/sales data and aggregate to monthly totals.

	Returns a DataFrame with a single column 'sales' indexed by month start.
	"""
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
	"""Handle wide weekly dataset like Sales_Transactions_Dataset_Weekly.

	- Uses W0..Wn columns as raw weekly sales for each product.
	- Sums across products to get a total weekly "sales" series.
	- Optionally uses corresponding "Normalized i" columns to build a
	  second feature capturing the average normalized weekly level
	  across products.

	Returns a DataFrame indexed by synthetic weekly dates with columns:
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

	# Create an arbitrary weekly date index; absolute dates are not
	# important for the LSTM, only the order and spacing. 2000-01-03
	# is a Monday.
	weekly_index = pd.date_range(
		start="2000-01-03", periods=len(weekly_cols_sorted), freq="W-MON"
	)

	sales_series = pd.Series(weekly_sum.values, index=weekly_index, name="sales")

	# Optional second feature: mean of per-product normalized values
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
		# Ensure we have the same number of normalized columns as weekly
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
	"""Load a sales time series from CSV or Excel.

	Preferred format (generic):
	- Columns: 'date' and 'sales' (any frequency). We resample to
	  monthly and return a DataFrame with a 'sales' column.

	Assignment convenience format:
	- Wide weekly dataset with W0..Wn and optional Normalized 0..n.
	  We aggregate to a single weekly time series of total sales and
	  an optional second feature 'norm'.
	"""
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


def make_lstm_sequences(values: np.ndarray, lookback: int = 12):
	"""Build (X, y) for LSTM with shape (samples, lookback, n_features).

	The target y is always the first feature (column 0), which we treat
	as the total sales value to forecast. Additional columns in
	`values` can be auxiliary features (e.g., mean normalized level).
	"""
	if values.ndim != 2:
		raise ValueError("values must have shape (n_samples, n_features)")

	n_samples, n_features = values.shape
	if n_samples <= lookback:
		raise ValueError("Not enough data to build sequences with given lookback")

	X, y = [], []
	for i in range(n_samples - lookback):
		X.append(values[i : i + lookback, :])
		# Target is the first feature at the prediction step
		y.append(values[i + lookback, 0])

	X = np.array(X, dtype="float32")
	y = np.array(y, dtype="float32").reshape(-1, 1)

	return X, y


def build_lstm_model(lookback: int, n_features: int) -> tf.keras.Model:
	"""Create the LSTM model with the requested architecture."""
	model = models.Sequential(
		[
			layers.Input(shape=(lookback, n_features)),
			layers.LSTM(64),
			layers.Dense(32, activation="relu"),
			layers.Dense(1),
		]
	)

	model.compile(optimizer="adam", loss="mse")
	return model


def _inverse_first_feature(scaled_col: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
	"""Inverse-transform the first feature using a multi-feature scaler.

	`scaled_col` is an array of shape (n_samples, 1) containing the
	scaled values of the first feature only. We construct dummy values
	for the remaining features (zeros), apply `scaler.inverse_transform`,
	and then return the first column in original units.
	"""
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


def train_model(
	input_path: str,
	model_out: str,
	epochs: int,
	lookback: int = 12,
	batch_size: int = 16,
	patience: int = 8,
):
	"""Train an LSTM on sales data and save artifacts.

	Supports both standard date/sales inputs and the assignment's wide
	weekly dataset. When the weekly dataset is used, the model sees two
	input features per time step: total weekly sales and mean normalized
	level across products.
	"""
	ts_df = load_sales_file(input_path)
	date_mode = ts_df.attrs.get("date_mode", "unknown")

	values = ts_df.values.astype("float32")
	n_total, n_features = values.shape
	n_val_steps = 12

	if n_total < lookback + n_val_steps:
		raise ValueError(
			"Time series is too short. Need at least lookback + 12 steps of data."
		)

	n_train = n_total - n_val_steps

	# Fit scaler on training portion only (all features)
	scaler = MinMaxScaler(feature_range=(0.0, 1.0))
	scaler.fit(values[:n_train])
	scaled_values = scaler.transform(values)

	# Build supervised sequences
	X_all, y_all = make_lstm_sequences(scaled_values, lookback=lookback)

	# Determine split point in sequence space so that
	# the last 12 steps (labels) form the validation set.
	# y_all[k] corresponds to original index lookback + k.
	seq_split_index = n_train - lookback
	if seq_split_index <= 0 or seq_split_index >= len(y_all):
		raise ValueError("Not enough history to create a separate validation set.")

	X_train, y_train = X_all[:seq_split_index], y_all[:seq_split_index]
	X_val, y_val = X_all[seq_split_index:], y_all[seq_split_index:]

	# Build and train model
	model = build_lstm_model(lookback, n_features=n_features)
	es = callbacks.EarlyStopping(
		monitor="val_loss", patience=patience, restore_best_weights=True
	)

	history = model.fit(
		X_train,
		y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[es],
		verbose=1,
	)

	# ------------------------------------------------------------------
	# Evaluate on validation set (in original sales units)
	# ------------------------------------------------------------------
	y_val_pred_scaled = model.predict(X_val, verbose=0)
	y_val_scaled = y_val

	y_val_pred = _inverse_first_feature(y_val_pred_scaled, scaler)
	y_val_true = _inverse_first_feature(y_val_scaled, scaler)

	diff = y_val_pred - y_val_true
	mae = float(np.mean(np.abs(diff)))
	rmse = float(np.sqrt(np.mean(diff ** 2)))

	non_zero = y_val_true != 0
	if np.any(non_zero):
		mape = float(
			np.mean(np.abs(diff[non_zero] / y_val_true[non_zero])) * 100.0
		)
	else:
		mape = None

	# Prepare output directory
	os.makedirs(model_out, exist_ok=True)

	# Save model in SavedModel format under lstm_saved_model/
	saved_model_dir = os.path.join(model_out, "lstm_saved_model")
	model.save(saved_model_dir)

	# Save scaler
	scaler_path = os.path.join(model_out, "scaler.pkl")
	try:
		import joblib

		joblib.dump(scaler, scaler_path)
	except Exception:  # pragma: no cover - serialization specifics
		import pickle

		with open(scaler_path, "wb") as f:
			pickle.dump(scaler, f)

	# Save metadata
	metadata_path = os.path.join(model_out, "metadata.json")
	metadata = {
		"lookback": lookback,
		"n_features": int(n_features),
		"n_total_months": int(n_total),
		"n_train_months": int(n_train),
		"n_val_months": int(n_val_steps),
		"loss_history": list(map(float, history.history.get("loss", []))),
		"val_loss_history": list(map(float, history.history.get("val_loss", []))),
		"val_mae": mae,
		"val_rmse": rmse,
		"val_mape": mape,
		"date_mode": date_mode,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"input_path": os.path.abspath(input_path),
		"saved_model_dir": os.path.abspath(saved_model_dir),
		"scaler_path": os.path.abspath(scaler_path),
	}
	with open(metadata_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)

	print(f"Training complete. Model artifacts saved to: {os.path.abspath(model_out)}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train an LSTM model for sales forecasting."
	)
	parser.add_argument(
		"--input",
		required=True,
		help=(
			"Path to input sales file (.csv or .xlsx). Either a date/sales file "
			"or the assignment's weekly dataset."
		),
	)
	parser.add_argument(
		"--model_out",
		required=True,
		help="Directory where the model, scaler, and metadata will be saved.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=50,
		help="Number of training epochs (default: 50)",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=16,
		help="Batch size for training (default: 16)",
	)
	parser.add_argument(
		"--patience",
		type=int,
		default=8,
		help="EarlyStopping patience in epochs (default: 8)",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	try:
		train_model(
			input_path=args.input,
			model_out=args.model_out,
			epochs=args.epochs,
			lookback=12,
			batch_size=args.batch_size,
			patience=args.patience,
		)
	except Exception as exc:
		print(f"Error: {exc}")
		raise SystemExit(1)


if __name__ == "__main__":
	main()
