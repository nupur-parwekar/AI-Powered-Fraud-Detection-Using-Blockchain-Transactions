from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Step 1: Load the saved model and scaler
model   = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")
le      = joblib.load("label_encoder.pkl")
print("Model loaded successfully!")

# Step 2: Create the FastAPI app
app = FastAPI(title="Fraud Detection API")

# Step 3: Define what a transaction looks like
class Transaction(BaseModel):
    amount: float
    hour: int
    merchant_category: str
    is_foreign: int
    customer_age: int

# Step 4: Home route — just to check API is running
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

# Step 5: Predict route — the main fraud detection endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert merchant_category text to number
    try:
        category_encoded = le.transform([transaction.merchant_category])[0]
    except:
        category_encoded = 0  # unknown category defaults to 0

    # Arrange features in correct order
    features = np.array([[
        transaction.amount,
        transaction.hour,
        category_encoded,
        transaction.is_foreign,
        transaction.customer_age
    ]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Get prediction and fraud probability
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "transaction_amount": transaction.amount,
        "merchant_category": transaction.merchant_category,
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "alert": "FRAUD DETECTED" if prediction == 1 else "Transaction looks safe"
    }

# Step 6: Metrics route — shows model info
@app.get("/metrics")
def metrics():
    return {
        "model": "Random Forest",
        "total_trees": 100,
        "features_used": ["amount", "hour", "merchant_category", "is_foreign", "customer_age"],
        "status": "active"
    }
