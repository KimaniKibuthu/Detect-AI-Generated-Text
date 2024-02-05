import pytest
from src.inference.model_inference import get_inference


# Define a fixture for sample data
@pytest.fixture
def sample_data():
    return "This is a sample input text."

def test_get_inference(sample_data):
    # Test getting inference with sample data
    predictions = get_inference(sample_data)

    # Check if the result is a list
    assert isinstance(predictions, list)

    # Check if the list contains valid elements
    for prediction in predictions:
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 1
