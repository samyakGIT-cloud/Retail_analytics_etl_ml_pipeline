import pandas as pd
import psycopg2

CSV_FILE = "cleaned_data.csv"

DB_CONFIG = {
    "host": "localhost",
    "database": "retail_db",
    "user": "postgres",
    "password": "3016"
}

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("CSV loaded successfully.")
    print("Columns found:", list(df.columns))
    return df

def clean_dataframe(df):
    df = df.dropna(subset=["Customer ID"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    #Total Price revenue Analysis 
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    print("Data cleaned successfully.")
    return df

def insert_data(df):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO retail_data
        (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country, TotalPrice)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    inserted_rows = 0
# For each row of a DataFrame comes in a loop
    for _, row in df.iterrows():
        cursor.execute(
            insert_query,
            (
                str(row["Invoice"]),
                str(row["StockCode"]),
                str(row["Description"]),
                int(row["Quantity"]),
                row["InvoiceDate"].to_pydatetime(),
                float(row["Price"]),
                float(row["Customer ID"]),
                str(row["Country"]),
                float(row["TotalPrice"])
            )
        )
        inserted_rows += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Inserted {inserted_rows} rows into PostgreSQL successfully.")

def main():
    print("Step 1: Loading CSV...")
    df = load_data(CSV_FILE)

    print("Step 2: Cleaning data...")
    df = clean_dataframe(df)

    print("Step 3: Inserting data into PostgreSQL...")
    insert_data(df)

    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    main()
