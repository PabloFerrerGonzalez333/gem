import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import numpy as np

from src.serving.app import app, ModelArtifacts

client = TestClient(app)


@pytest.fixture
def mock_artifacts():
    """Mock the model artifacts to avoid requiring a trained model for basic API tests."""
    mock_model = MagicMock()
    # Let's say predict_proba returns array of probabilities for 2 classes
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

    return ModelArtifacts(model=mock_model, feature_cols=["feature1", "feature2"])


@patch("src.serving.app.load_artifacts")
def test_health(mock_load, mock_artifacts):
    mock_load.return_value = mock_artifacts
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "n_features": 2}


@patch("src.serving.app.load_artifacts")
def test_schema(mock_load, mock_artifacts):
    mock_load.return_value = mock_artifacts
    response = client.get("/schema")
    assert response.status_code == 200
    assert response.json() == {"feature_columns": ["feature1", "feature2"]}


@patch("src.serving.app.load_artifacts")
def test_predict(mock_load, mock_artifacts):
    mock_load.return_value = mock_artifacts

    payload = {
        "records": [
            {"feature1": 1.0, "feature2": 0.0},
            {"feature1": 0.0, "feature2": 1.0},
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"] == [1, 0]  # Predictions: 0.9 -> 1, 0.2 -> 0
