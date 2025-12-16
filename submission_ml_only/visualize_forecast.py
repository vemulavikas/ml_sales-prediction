import argparse
import os

import matplotlib.pyplot as plt

from predict_lstm import forecast_to_dataframe


def plot_and_show(df, out_image: str | None = None):
    """Plot a simple bar chart from a forecast DataFrame.

    DataFrame must contain 'date' (datetime) and 'forecast' (numeric).
    """
    dates = df["date"]
    values = df["forecast"]

    plt.figure(figsize=(10, 5))
    plt.bar(dates.dt.strftime("%Y-%m"), values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Forecasted Sales")
    plt.title("Forecasted Monthly Sales")
    plt.tight_layout()

    if out_image:
        out_dir = os.path.dirname(os.path.abspath(out_image)) or "."
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_image)
        print(f"Chart saved to: {os.path.abspath(out_image)}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize LSTM forecasts: prints a grid and shows/saves a chart. "
            "This is a helper around predict_lstm.py for local ML visualization."
        )
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory with lstm_saved_model/, scaler.pkl, metadata.json.",
    )
    parser.add_argument(
        "--history",
        required=True,
        help="History file (.csv or .xlsx); can be the weekly assignment dataset.",
    )
    parser.add_argument(
        "--predict_months",
        type=int,
        default=12,
        help="Number of future months to visualize (default: 12).",
    )
    parser.add_argument(
        "--out_image",
        help="Optional path to save the chart image (e.g. forecast.png).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get forecast as DataFrame (date, forecast)
    df = forecast_to_dataframe(
        model_dir=args.model_dir,
        history_path=args.history,
        predict_months=args.predict_months,
    )

    # Print grid (table) to console
    print("\nForecast grid (first rows):")
    print(df.head(args.predict_months).to_string(index=False))

    # Plot bar chart
    plot_and_show(df, out_image=args.out_image)


if __name__ == "__main__":
    main()
