# import pytest
# from unittest.mock import MagicMock, patch
# import pandas as pd
# from src.inference.model_inference import get_inference
# from src.utils import load_config
# # Define the test input data

# @pytest.fixture
# def config():
#     return load_config()

# @pytest.fixture
# def mock_data():
#     return "Test data"

# @patch('mlflow.pyfunc.load_model')
# @patch('builtins.open')
# @patch('src.data_handling.build_features.vectorize_data')
# @patch('src.data_handling.build_features.tokenize_data')
# def test_get_inference_success(mock_tokenize, mock_vectorize, mock_open, mock_load_model, mock_data, config):
#     # Mock dependencies
#     mock_open.return_value.__enter__.return_value = MagicMock()
#     mock_load_model.return_value.predict.return_value = [1]

#     # Call the function with the test input data
#     result = get_inference(mock_data)

#     # Assert that the function returns the expected result
#     assert result == [1]

#     # Assert that dependencies were called with the correct arguments
#     mock_open.assert_called_once_with("vectorizer.pkl", 'rb')
#     mock_tokenize.assert_called_once_with(pd.DataFrame(columns=['text'], data=[mock_data]), "sentpiece.model")
#     mock_vectorize.assert_called_once_with(mock_open.return_value.__enter__.return_value, mock_tokenize.return_value["text_spm"])
#     mock_load_model.assert_called_once_with("model_uri")

#     # Assert that loaded model's predict method was called
#     mock_load_model.return_value.predict.assert_called_once_with(mock_vectorize.return_value)

# @patch('mlflow.pyfunc.load_model')
# def test_get_inference_failure(mock_load_model, mock_data, config):
#     # Mock load_model to raise an exception
#     mock_load_model.side_effect = Exception("Model load failed")

#     # Call the function and assert that it raises the expected exception
#     with pytest.raises(Exception) as context:
#         get_inference(mock_data)

#     # Assert that the exception message matches the expected one
#     assert 'Model load failed' in str(context.value)
