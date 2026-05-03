import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("cleaned_data.csv")

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

sales = daily_sales["TotalPrice"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)

# Create sequences
lookback = 30

X = []
y = []

for i in range(lookback, len(sales_scaled)):
    X.append(sales_scaled[i-lookback:i])
    y.append(sales_scaled[i])

X = np.array(X)
y = np.array(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# LSTM Model
class SalesLSTM(nn.Module):
    def __init__(self):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = SalesLSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 50

for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Forecast next 30 days
last_sequence = sales_scaled[-lookback:]
future_predictions = []

current_seq = torch.tensor(last_sequence.reshape(1, lookback, 1), dtype=torch.float32)

for _ in range(30):
    pred = model(current_seq)
    future_predictions.append(pred.item())

    new_value = torch.tensor([[[pred.item()]]], dtype=torch.float32)
    current_seq = torch.cat((current_seq[:, 1:, :], new_value), dim=1)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

future_dates = pd.date_range(
    start=daily_sales["Date"].max() + pd.Timedelta(days=1),
    periods=30
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "LSTM_PredictedSales": future_predictions.flatten()
})

print("LSTM 30-Day Forecast:")
print(forecast_df)

forecast_df.to_csv("lstm_30_days_forecast.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["Date"], daily_sales["TotalPrice"], label="Actual Sales")
plt.plot(forecast_df["Date"], forecast_df["LSTM_PredictedSales"], label="LSTM Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("LSTM 30-Day Sales Forecast")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_forecast_chart.png")
plt.show()

print("LSTM forecasting completed successfully.")
print("Files saved: lstm_30_days_forecast.csv and lstm_forecast_chart.png")
