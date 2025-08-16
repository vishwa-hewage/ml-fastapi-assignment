from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Map target numbers to species names
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

app = FastAPI(
    title="Iris Flower Classification API",
    description="Predict iris species using a trained RandomForest model",
    version="1.0"
)

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput):
    try:
        X = np.array([[input_data.sepal_length,
                       input_data.sepal_width,
                       input_data.petal_length,
                       input_data.petal_width]])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        return PredictionOutput(
            prediction=species_map[pred],
            confidence=float(np.max(probs))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_classes": list(species_map.values())
    }
