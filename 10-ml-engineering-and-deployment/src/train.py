
from pathlib import Path
import json

import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "digits_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def main():
    # 1. Load the dataset
    digits = load_digits()

    X = digits.data
    y = digits.target

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Build the ML pipeline
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    # 4. Train the model
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate the model
    train_predictions = model_pipeline.predict(X_train)
    test_predictions = model_pipeline.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )
    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    # 6. Create the artifacts directory
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 7. Save the complete trained pipeline
    joblib.dump(model_pipeline, MODEL_PATH)

    # 8. Save evaluation metrics
    metrics = {
        "dataset": "scikit-learn Digits",
        "random_state": 42,
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "input_features": int(X.shape[1]),
        "num_classes": int(len(set(y))),
        "training_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 9. Print a summary
    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
