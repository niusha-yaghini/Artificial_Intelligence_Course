
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "digits_pipeline.joblib"
)


def load_model():
    """Load the trained pipeline from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run the training script first."
        )

    return joblib.load(MODEL_PATH)


def predict_digit(features, model):
    """
    Predict a handwritten digit from 64 raw pixel values.

    Parameters
    ----------
    features : array-like
        A one-dimensional array of 64 pixel intensities.
        Expected values are in the range [0, 16].

    model : sklearn.pipeline.Pipeline
        A previously loaded, trained pipeline.

    Returns
    -------
    dict
        Predicted digit and maximum predicted class probability.
    """

    features = np.asarray(features, dtype=np.float64)

    if features.shape != (64,):
        raise ValueError(
            f"Expected 64 features, got shape {features.shape}."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Input features must be finite numbers."
        )

    if np.any(features < 0) or np.any(features > 16):
        raise ValueError(
            "Pixel intensities must be between 0 and 16."
        )

    sample = features.reshape(1, 64)

    probabilities = model.predict_proba(sample)[0]
    classes = model.named_steps["classifier"].classes_

    best_index = int(np.argmax(probabilities))

    return {
        "predicted_digit": int(classes[best_index]),
        "confidence": float(probabilities[best_index]),
    }
