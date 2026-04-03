"""
Prediction API (Telco Churn) using FastAPI.

Exposes the endpoints:

- ``GET /health``: Service status and number of expected *features*.
- ``GET /schema``: Input columns required by the model.
- ``POST /predict``: Batch prediction (probabilities and labels).

Notes
-----
- The API loads artifacts from ``models/best`` by default.
- The input payload must include a ``records`` field with a list of
  dictionaries (one row per dictionary).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError


# ==========================
#         Config
# ==========================

MODEL_DIR = Path("models/best")

app = FastAPI(title="Telco Churn API", version="1.0.0")
SPHINX_HTML_DIR = Path("docs/build/html")
SPHINX_HTML_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/documentation",
    StaticFiles(directory=str(SPHINX_HTML_DIR), html=True),
    name="sphinx-docs",
)


# ==========================
#      Model artifacts
# ==========================


class ModelArtifacts(BaseModel):
    """
    Container for model artifacts.

    :param model: sklearn/pipeline estimator loaded with ``joblib``.
    :param feature_cols: Ordered list of expected columns.
    """

    model: Any
    feature_cols: List[str]


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts:
    """
    Loads and caches the model artifacts from ``MODEL_DIR``.

    :return: Artifacts including the model and feature columns.
    :rtype: ModelArtifacts
    :raises FileNotFoundError: If any expected artifact is missing.
    :raises Exception: If model deserialization fails.
    """
    model_path = MODEL_DIR / "model.joblib"
    feats_path = MODEL_DIR / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing: {feats_path}")

    model = joblib.load(model_path)
    feature_cols: List[str] = json.loads(feats_path.read_text())

    if not isinstance(feature_cols, list) or not all(
        isinstance(c, str) for c in feature_cols
    ):
        raise ValueError("feature_columns.json must be list[str].")

    return ModelArtifacts(model=model, feature_cols=feature_cols)


# ==========================
#     Pydantic Schemas
# ==========================


class PredictRequest(BaseModel):
    """
    Batch prediction request.

    :param records: List of rows; each row is a dict {feature: value}.
    """

    records: Sequence[Dict[str, Any]] = Field(min_length=1)


class PredictResponse(BaseModel):
    """
    Prediction response.

    :param predictions: Labels {0,1} thresholded at 0.5.
    :param probabilities: Probabilities of the positive class.
    """

    predictions: List[int]
    probabilities: List[float]


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    n_features: int


class SchemaResponse(BaseModel):
    """Response containing the input *feature* schema."""

    feature_columns: List[str]


# ==========================
#       Utilities
# ==========================


def _align_payload(
    rows: Sequence[Dict[str, Any]], feature_cols: List[str]
) -> pd.DataFrame:
    """
    Aligns and validates the input payload against ``feature_cols``.

    - Adds missing columns with 0.
    - Reorders columns according to training order.
    - Discards extra columns (strict policy).

    :param rows: Raw rows from the payload.
    :param feature_cols: Expected list of columns.
    :return: DataFrame ready for ``predict_proba``.
    :raises ValueError: If ``rows`` cannot be converted to a DataFrame.
    """
    try:
        df = pd.DataFrame(rows)
    except Exception as exc:  # noqa: E722  (caught for clarity)
        raise ValueError("Invalid 'records' format for DataFrame.") from exc

    # Fill missing columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Strict ordering and discarding of extra columns
    df = df[[c for c in feature_cols]]
    return df


# ==========================
#         Endpoints
# ==========================


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Returns the service status and the number of required *features*.

    :return: Object containing ``status`` and ``n_features``.
    """
    feats = load_artifacts().feature_cols
    return HealthResponse(status="ok", n_features=len(feats))


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    """
    Returns the input schema expected by the model.

    :return: Ordered list of *feature* columns.
    """
    feats = load_artifacts().feature_cols
    return SchemaResponse(feature_columns=feats)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """
    Predicts labels and probabilities for the rows in ``payload.records``.

    :param payload: Request with a list of records.
    :return: Probabilities (positive class) and thresholded predictions.
    :raises HTTPException: If a prediction error occurs.
    """
    try:
        artifacts = load_artifacts()
        X = _align_payload(payload.records, artifacts.feature_cols)

        # Index of the positive class; conventionally we use column 1
        proba = artifacts.model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return PredictResponse(
            predictions=preds.tolist(),
            probabilities=proba.astype(float).tolist(),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        ) from exc


# Run with:
# poetry run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
