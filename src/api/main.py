from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# Load the trained model from MLflow
model_uri = "models:/best_model/Production"  # replace with your registered model path
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI(title="BNPL Credit Risk API")

class CustomerData(BaseModel):
    CustomerId: int
    Feature1: float
    Feature2: float
    # Add all features used by the model

@app.post("/predict")
def predict_risk(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    # Make prediction
    risk_prob = model.predict_proba(df)[:,1][0]  # probability of high risk
    return {"CustomerId": data.CustomerId, "risk_probability": risk_prob}
