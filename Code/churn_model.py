import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("cleaned_data.csv")

# Basic preprocessing
df.dropna(subset=["Customer ID"], inplace=True)

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True, errors="coerce")
df = df.dropna(subset=["InvoiceDate"])
# Create recency feature (days since last purchase)
latest_date = df["InvoiceDate"].max()
recency = df.groupby("Customer ID")["InvoiceDate"].max().reset_index()
recency["Recency"] = (latest_date - recency["InvoiceDate"]).dt.days

# Create frequency feature
frequency = df.groupby("Customer ID")["Invoice"].nunique().reset_index()
frequency.columns = ["Customer ID", "Frequency"]

# Create monetary feature
monetary = df.groupby("Customer ID")["TotalPrice"].sum().reset_index()

# Merge all
rfm = recency.merge(frequency, on="Customer ID")
rfm = rfm.merge(monetary, on="Customer ID")

# Create churn label (simple logic)
rfm["Churn"] = rfm["Recency"].apply(lambda x: 1 if x > 90 else 0)

# Features
X = rfm[["Recency", "Frequency", "TotalPrice"]]
y = rfm["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
