import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException
from pydantic import BaseModel
from src.utils import load_config
from src.pipelines.full_pipeline import main
from src.inference.model_inference import get_inference

app = FastAPI()
config = load_config()

class PredictionInput(BaseModel):
    text: str

class TrainingInput(BaseModel):
    train: bool

# Define the path to the logged model
logged_model = config["models"]["model_uri"]

def load_mlflow_model(model_uri: str):
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except MlflowException as e:
        raise HTTPException(status_code=500, detail=f"Error loading MLflow model: {str(e)}")

loaded_model = load_mlflow_model(logged_model)

@app.get('/predict/')
async def predict(text: str):
    try:
        predictions = get_inference(text)
        return JSONResponse(content={
            'predictions': predictions.tolist(),
            'predictions_class': 'AI' if predictions[0] == 1 else 'Not AI'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def train(input_data: TrainingInput):
    try:
        if input_data.train:
            main(config)
        return {'status': 'Training completed successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
