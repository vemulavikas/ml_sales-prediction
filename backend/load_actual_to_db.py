import pandas as pd
from db import get_connection

CSV_PATH = "../Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv"

def load_actual():
    df = pd.read_csv(CSV_PATH)

    weekly_cols = [c for c in df.columns if c.startswith("W")]
    weekly_sum = df[weekly_cols].sum()

    conn = get_connection()
    cur = conn.cursor()

    year = 2024
    month = 1

    for value in weekly_sum[:12]:
        cur.execute(
            "INSERT INTO actual_sales (month, year, amount) VALUES (%s,%s,%s)",
            (month, year, int(value))
        )
        month += 1

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    load_actual()
