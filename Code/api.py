from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="NeuralRetail API")

@app.get("/")
def home():
    return {"message": "NeuralRetail API is running"}

@app.get("/forecast/baseline")
def baseline_forecast():
    df = pd.read_csv("sales_forecast_30_days.csv")
    return df.to_dict(orient="records")

@app.get("/forecast/prophet")
def prophet_forecast():
    df = pd.read_csv("prophet_30_days_forecast.csv")
    return df.to_dict(orient="records")

@app.get("/forecast/lstm")
def lstm_forecast():
    df = pd.read_csv("lstm_30_days_forecast.csv")
    return df.to_dict(orient="records")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
