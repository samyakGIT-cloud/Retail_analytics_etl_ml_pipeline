import streamlit as st
import pandas as pd
import psycopg2
import os  #For docker 

st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")

conn = psycopg2.connect(
    host=DB_HOST,
    database="retail_db",
    user="postgres",
    password="3016",
    port="5432"
)

# Load data from PostgreSQL
df = pd.read_sql("SELECT * FROM retail_data;", conn)

# Title(UI Heading)
st.title("Retail Analytics Dashboard (ETL + ML Forecasting)")

# Sidebar filter
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["country"].dropna().unique().tolist())
)

if selected_country != "All":
    df = df[df["country"] == selected_country]

# KPI Metrics
total_revenue = df["totalprice"].sum()
total_customers = df["customerid"].nunique()
total_orders = df["invoice"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"{total_revenue:,.2f}")
col2.metric("Total Customers", total_customers)
col3.metric("Total Orders", total_orders)

st.markdown("---") # adds a line break in the UI 

# Top Products
st.subheader("Top 10 Products by Quantity")
top_products = (
    df.groupby("description")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_products)

# Top Customers
st.markdown("---")
st.subheader("Top 10 Customers by Revenue")
top_customers = (
    df.groupby("customerid")["totalprice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_customers)

# Revenue by Country
st.markdown("---")
st.subheader("Revenue by Country")
country_revenue = (
    df.groupby("country")["totalprice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(country_revenue)

# Sales Trend
st.markdown("---")
st.subheader("Sales Trend")
df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
daily_sales = df.groupby(df["invoicedate"].dt.date)["totalprice"].sum()
st.line_chart(daily_sales)

# Prophet Forecast - Best Model
st.markdown("---")
st.subheader("Prophet 30-Day Sales Forecast")

prophet_df = pd.read_csv("/app/prophet_30_days_forecast.csv")
st.dataframe(prophet_df)

st.image("prophet_forecast_chart.png", caption="Prophet Forecast: Actual + Future Sales")

# Baseline Forecast - Linear Regression
st.markdown("---")
st.subheader("30-Day Sales Forecast (Baseline Model)")

forecast_df = pd.read_csv("/app/sales_forecast_30_days.csv")
st.dataframe(forecast_df)

st.image("sales_forecast_chart.png", caption="Baseline Forecast")

# LSTM Forecast - Advanced Model
st.markdown("---")
st.subheader("LSTM 30-Day Sales Forecast (Advanced)")

lstm_df = pd.read_csv("/app/lstm_30_days_forecast.csv")
st.dataframe(lstm_df)

st.image("lstm_forecast_chart.png", caption="LSTM Forecast")




