import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

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

# Prophet needs columns: ds and y
prophet_df = daily_sales.rename(columns={
    "Date": "ds",
    "TotalPrice": "y"
})

prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

# Model
model = Prophet()
model.fit(prophet_df)

# Future 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Save forecast
forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
forecast_output.to_csv("prophet_30_days_forecast.csv", index=False)

print("Prophet 30-Day Forecast:")
print(forecast_output)

# Plot
fig = model.plot(forecast)
plt.title("Prophet 30-Day Sales Forecast")
plt.tight_layout()
plt.savefig("prophet_forecast_chart.png")
plt.show()

print("Prophet forecasting completed successfully.")
print("Files saved: prophet_30_days_forecast.csv and prophet_forecast_chart.png")
