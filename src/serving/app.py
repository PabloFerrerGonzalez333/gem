"""
API de predicción (Telco Churn) con FastAPI.

Expone los endpoints:

- ``GET /health``: estado del servicio y nº de *features* esperadas.
- ``GET /schema``: columnas de entrada requeridas por el modelo.
- ``POST /predict``: predicción por lotes (probabilidades y etiquetas).

Notas
-----
- La API carga los artefactos desde ``models/best`` por defecto.
- El payload de entrada debe incluir un campo ``records`` con una lista de
  diccionarios (una fila por diccionario).
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
    Contenedor de artefactos del modelo.

    :param model: Estimador sklearn/pipeline cargado con ``joblib``.
    :param feature_cols: Lista ordenada de columnas esperadas.
    """

    model: Any
    feature_cols: List[str]


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts:
    """
    Carga y cachea los artefactos del modelo desde ``MODEL_DIR``.

    :return: Artefactos con modelo y columnas de *features*.
    :rtype: ModelArtifacts
    :raises FileNotFoundError: Si falta algún artefacto requerido.
    :raises Exception: Si falla la deserialización del modelo.
    """
    model_path = MODEL_DIR / "model.joblib"
    feats_path = MODEL_DIR / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe: {model_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"No existe: {feats_path}")

    model = joblib.load(model_path)
    feature_cols: List[str] = json.loads(feats_path.read_text())

    if not isinstance(feature_cols, list) or not all(
        isinstance(c, str) for c in feature_cols
    ):
        raise ValueError("feature_columns.json debe ser list[str].")

    return ModelArtifacts(model=model, feature_cols=feature_cols)


# ==========================
#     Esquemas Pydantic
# ==========================


class PredictRequest(BaseModel):
    """
    Petición de predicción por lotes.

    :param records: Lista de filas; cada fila es un dict {feature: valor}.
    """

    records: Sequence[Dict[str, Any]] = Field(min_length=1)


class PredictResponse(BaseModel):
    """
    Respuesta de predicción.

    :param predictions: Etiquetas {0,1} umbralizadas a 0.5.
    :param probabilities: Probabilidades de clase positiva.
    """

    predictions: List[int]
    probabilities: List[float]


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str
    n_features: int


class SchemaResponse(BaseModel):
    """Respuesta con el esquema de *features* de entrada."""

    feature_columns: List[str]


# ==========================
#       Utilidades
# ==========================


def _align_payload(
    rows: Sequence[Dict[str, Any]], feature_cols: List[str]
) -> pd.DataFrame:
    """
    Alinea y valida el payload de entrada contra ``feature_cols``.

    - Añade columnas ausentes con 0.
    - Ordena columnas según el entrenamiento.
    - Descarta columnas extra (política estricta).

    :param rows: Filas crudas del payload.
    :param feature_cols: Lista esperada de columnas.
    :return: DataFrame listo para ``predict_proba``.
    :raises ValueError: Si ``rows`` no es convertible a DataFrame.
    """
    try:
        df = pd.DataFrame(rows)
    except Exception as exc:  # noqa: E722  (se captura para claridad)
        raise ValueError("Formato de 'records' inválido para DataFrame.") from exc

    # Completar columnas faltantes
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Orden estricto y descarte de columnas extra
    df = df[[c for c in feature_cols]]
    return df


# ==========================
#         Endpoints
# ==========================


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Devuelve el estado del servicio y el número de *features* requeridas.

    :return: Objeto con ``status`` y ``n_features``.
    """
    feats = load_artifacts().feature_cols
    return HealthResponse(status="ok", n_features=len(feats))


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    """
    Devuelve el esquema de entrada esperado por el modelo.

    :return: Lista ordenada de columnas de *features*.
    """
    feats = load_artifacts().feature_cols
    return SchemaResponse(feature_columns=feats)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """
    Predice etiquetas y probabilidades para las filas de ``payload.records``.

    :param payload: Petición con lista de registros.
    :return: Probabilidades (clase positiva) y predicciones umbralizadas.
    :raises HTTPException: Si ocurre un error de predicción.
    """
    try:
        artifacts = load_artifacts()
        X = _align_payload(payload.records, artifacts.feature_cols)

        # Índice de la clase positiva; por convenio usamos la columna 1
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
            detail=f"Error en predicción: {exc}",
        ) from exc


# Ejecuta con:
# poetry run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
