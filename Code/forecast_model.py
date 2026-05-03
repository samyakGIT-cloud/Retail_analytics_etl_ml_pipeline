import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Retail Forecasting")

# Load data
df = pd.read_csv("cleaned_data.csv")

# Date fix
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["InvoiceDate", "TotalPrice"])

# Daily sales
df["Date"] = df["InvoiceDate"].dt.date
daily_sales = df.groupby("Date")["TotalPrice"].sum().reset_index()

daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
daily_sales = daily_sales.sort_values("Date")

# Create numeric day column
daily_sales["DayNumber"] = range(len(daily_sales))

X = daily_sales[["DayNumber"]]
y = daily_sales["TotalPrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast next 30 days
last_day = daily_sales["DayNumber"].max()

future_days = pd.DataFrame({
    "DayNumber": range(last_day + 1, last_day + 31)
})

future_sales = model.predict(future_days)

future_dates = pd.date_range(
    start=daily_sales["Date"].max() + pd.Timedelta(days=1),
    periods=30
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "PredictedSales": future_sales
})

print("Next 30 Days Forecast:")
print(forecast_df)

# Save forecast CSV
forecast_df.to_csv("sales_forecast_30_days.csv", index=False)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["Date"], daily_sales["TotalPrice"], label="Actual Sales")
plt.plot(forecast_df["Date"], forecast_df["PredictedSales"], label="Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("30-Day Sales Forecast - Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast_chart.png")
plt.show()

print("Forecast completed successfully.")
print("Files saved: sales_forecast_30_days.csv and sales_forecast_chart.png")

# MLflow tracking
with mlflow.start_run(run_name="Linear Regression Forecast"):
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("forecast_horizon_days", 30)
    mlflow.log_param("features", "DayNumber")
    mlflow.log_param("target", "TotalPrice")

    mlflow.log_metric("training_rows", len(daily_sales))
    mlflow.log_metric("forecast_days", 30)

    mlflow.log_artifact("sales_forecast_30_days.csv")
    mlflow.log_artifact("sales_forecast_chart.png")

print("MLflow tracking completed successfully.")