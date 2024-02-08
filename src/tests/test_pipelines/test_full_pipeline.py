# import pytest
# from unittest.mock import Mock, patch
# from src.pipelines.full_pipeline import main
# from src.utils import load_config
# from logs import logger


# # Import load_config at the beginning of the test file
# from src.utils import load_config

# # Mock the mlflow module
# mlflow = Mock()

# # Mock the MlflowClient class
# MlflowClient = Mock()

# # Define a fixture for sample configuration
# @pytest.fixture
# def sample_config():
#     return load_config()

# def test_main(sample_config):
#     # Mock the load_config function
#     with patch('src.utils.load_config', return_value=sample_config) as mock_load_config:
#         # Mock the data_pipeline function
#         with patch('src.pipelines.data_pipeline.data_pipeline') as mock_data_pipeline:
#             # Mock the modelling_pipeline function
#             with patch('src.pipelines.modelling_pipeline.modelling_pipeline') as mock_modelling_pipeline:
#                 # Mock the logger.info function
#                 with patch('logs.logger.info') as mock_logger_info:
#                     # Call the main function
#                     main(sample_config)


#     # Check if the data_pipeline function was called exactly once
#     mock_data_pipeline.assert_called_once()

    # # Check if the modelling_pipeline function was called exactly once
    # mock_modelling_pipeline.assert_called_once()

    # # Check if the logger.info function was called exactly once
    # mock_logger_info.assert_called_once()

