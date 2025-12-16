import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="sales_forecast_db",
        user="postgres",
        password="project",
        port="5432"
    )
