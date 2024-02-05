
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.get("/predict/?text=SomeText")
    assert response.status_code == 200
    result = response.json()
    assert 'predictions' in result
    assert 'predictions_class' in result
    assert result['predictions_class'] in ['AI', 'Not AI']

def test_train_endpoint():
    response = client.post("/train", json={"train": True})
    assert response.status_code == 200
    result = response.json()
    assert 'status' in result
    assert result['status'] == 'Training completed successfully'
