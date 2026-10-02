"""
Simple FastAPI app serving a text classification scikit-learn model.

Run locally:
    python -m uvicorn main:app --reload

Open Swagger UI:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model.pkl"
)


app = FastAPI(
    title="Getting Started with ML in Production API",
    description="A minimal text classification prediction API built for the workshop.",
    version="1.0.0",
)


# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text that the machine learning model should classify."
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/")
def root():
    return {
        "message": "Workshop ML API is running. See /docs for usage."
    }


@app.get("/health")
def health():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check that model.pkl exists."
        )

    # The model contains a TfidfVectorizer,
    # so it expects text rather than numerical features.
    pred = int(model.predict([request.text])[0])

    return PredictionResponse(
        prediction=pred,
        class_name=f"class_{pred}"
    )