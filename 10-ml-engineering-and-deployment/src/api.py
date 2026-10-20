
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict_digit


# Create the API application
# This line creates an object from the FastAPI application. The Uvicorn web server will later execute this object.
app = FastAPI(
    title="Digits Classification API",
    description="Predict handwritten digits using a trained ML pipeline.",
    version="1.0.0",
)


# Load the model once when the API module is initialized
model = load_model()


class PredictionRequest(BaseModel):
    """Request body containing 64 raw pixel intensities."""

    features: list[float]


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    predicted_digit: int
    confidence: float


@app.get("/")
def root():
    return {
        "message": "Digits Classification API",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    # Validate the number of features
    if len(request.features) != 64:
        raise HTTPException(
            status_code=422,
            detail="Expected exactly 64 features.",
        )

    try:
        result = predict_digit(request.features, model)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return result
