from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import mlflow.pyfunc
import pandas as pd
from typing import List
from pydantic import BaseModel
from src.utils import load_config
from src.pipelines.full_pipeline import main
from src.inference.model_inference import get_inference

app = FastAPI()
config = load_config()

# Define the path to the logged model
logged_model = config["models"]["model_uri"]

# Load the model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

class PredictionInput(BaseModel):
    text: str

class TrainingInput(BaseModel):
    train: bool


@app.get('/predict/{text}')
async def predict(text: str):
    try:
        # Perform inference using the loaded model
        predictions = get_inference(text)

        # Return the predictions as JSON
        return JSONResponse(content={
            'predictions': predictions.tolist(),
            'predictions_class': 'AI' if predictions[0] == 1 else 'Not AI'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def train(input_data: TrainingInput):
    try:
        if input_data:
            main(config)

        return {'status': 'Training completed successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
