from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Perform model inference here and return the prediction
    prediction = f"Prediction for input: {request.input}"
    return PredictionResponse(prediction=prediction)
