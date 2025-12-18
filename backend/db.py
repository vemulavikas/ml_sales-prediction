import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST") or "localhost"
    database = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME") or "sales_forecast_db"
    user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER") or "postgres"
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD") or "project"
    port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT") or "5432"

    # Many AWS RDS Postgres setups enforce SSL (rds.force_ssl). If we connect
    # without SSL, Postgres rejects it with a pg_hba "no encryption" error.
    sslmode = (
        os.getenv("POSTGRES_SSLMODE")
        or os.getenv("DB_SSLMODE")
        or ("require" if host not in {"localhost", "127.0.0.1"} else None)
    )

    kwargs = {
        "host": host,
        "database": database,
        "user": user,
        "password": password,
        "port": port,
    }
    if sslmode:
        kwargs["sslmode"] = sslmode

    return psycopg2.connect(**kwargs)
