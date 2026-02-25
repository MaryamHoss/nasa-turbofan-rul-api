from fastapi import FastAPI, HTTPException, status, Request
import joblib
from contextlib import asynccontextmanager
import torch
import numpy as np
from app.schemas import PredictRequest, PredictResponse
from app.utils import transform_window, predict_output, LSTM_model
from pathlib import Path
from app.config import settings

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"



@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.scaler = joblib.load(MODEL_DIR / settings.scaler_file_name)
    app.state.model = LSTM_model(input_size=21, hidden_size=64, output_size=1)
    app.state.model.load_state_dict(torch.load(MODEL_DIR / settings.model_file_name, map_location="cpu"))
    app.state.model.eval()

    yield

    del app.state.scaler
    del app.state.model


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "RUL Prediction API is running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, client_input: PredictRequest):

    window = np.array(client_input.window, dtype=np.float32)
    #Validate 2D shape
    if window.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Window must be a 2D array"
        )
    #Validate dimensions match model
    if window.shape != (settings.window_length, settings.num_features):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Window must have shape ({settings.window_length}, {settings.num_features})"
        )

    try:
        scaler = request.app.state.scaler
        model = request.app.state.model
        window = np.array(client_input.window, dtype=np.float32)
        transformed_window = transform_window(scaler, window, settings.sensor_columns)
        predicted_RUL = predict_output(model, transformed_window)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


    return PredictResponse(
        engine_id=client_input.engine_id,
        predicted_RUL=predicted_RUL
    )





