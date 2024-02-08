import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

@pytest.fixture(scope="module")
def sample_text():
    return "SomeText"

@pytest.fixture(scope="module")
def mock_training():
    # Mock the training process here if necessary
    yield
    # Clean up after the test if needed

# def test_predict_endpoint(sample_text):
#     response = client.get(f"/predict/?text={sample_text}")
#     assert response.status_code == 200
#     result = response.json()
#     assert 'predictions' in result
#     assert 'predictions_class' in result
#     assert result['predictions_class'] in ['AI', 'Not AI']

def test_train_endpoint(mock_training):
    response = client.post("/train", json={"train": True})
    assert response.status_code == 200
    result = response.json()
    assert 'status' in result
    assert result['status'] == 'Training completed successfully'
