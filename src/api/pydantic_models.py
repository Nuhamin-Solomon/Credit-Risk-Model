from pydantic import BaseModel

class CustomerData(BaseModel):
    CustomerId: int
    Feature1: float
    Feature2: float
    # Add all features

class PredictionResponse(BaseModel):
    CustomerId: int
    risk_probability: float
