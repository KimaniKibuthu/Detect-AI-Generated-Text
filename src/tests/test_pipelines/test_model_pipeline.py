# import pytest
# from unittest.mock import patch, MagicMock, Mock
# from src.pipelines.modelling_pipeline import train_and_evaluate_model
# from mlflow.tracking import MlflowClient

# @pytest.fixture
# def mlflow_client():
#     # Create a mock MLflow client
#     return Mock(spec=MlflowClient)

# @pytest.fixture
# def mock_data():
#     # Create mock data
#     return {
#         'X_train': [[1, 2], [3, 4]],
#         'X_test': [[5, 6], [7, 8]],
#         'y_train': [0, 1],
#         'y_test': [0, 1]
#     }


# # Define mock objects for external dependencies
# mock_X_train = Mock()
# mock_y_train = Mock()
# mock_X_validation = Mock()
# mock_y_validation = Mock()
# mock_hyperparameter_tune = False
# mock_model = Mock()
# mock_roc = 0.85  # Mock ROC score

# @patch('src.pipelines.modelling_pipeline.hyperparameter_tuning')
# @patch('src.pipelines.modelling_pipeline.train_model')
# @patch('src.pipelines.modelling_pipeline.evaluate_and_log_metrics')
# @patch('src.pipelines.modelling_pipeline.log_model')
# def test_train_and_evaluate_model(mock_log_model, mock_evaluate_and_log_metrics, mock_train_model, mock_hyperparameter_tuning):
#     # Mock the return values of external functions
#     mock_hyperparameter_tuning.return_value = {'param1': 'value1', 'param2': 'value2'}
#     mock_train_model.return_value = mock_model
#     mock_evaluate_and_log_metrics.return_value = None  # No need to return anything for this mock

#     # Call the function under test
#     model, roc = train_and_evaluate_model(mock_X_train, mock_y_train, mock_X_validation, mock_y_validation, mock_hyperparameter_tune)

#     # Assert that the function behaves as expected
#     assert model == mock_model
#     assert roc == mock_roc
