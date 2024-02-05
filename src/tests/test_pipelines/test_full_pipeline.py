import pytest
from unittest.mock import Mock, patch
from src.pipelines.data_pipeline import data_pipeline
from src.pipelines.modelling_pipeline import modelling_pipeline
from src.pipelines.full_pipeline import main
from src.utils import load_config
from logs import logger

# Assuming your modules are in src.pipelines

# Mock the mlflow module
mlflow = Mock()

# Mock the MlflowClient class
MlflowClient = Mock()

# Define a fixture for sample configuration
@pytest.fixture
def sample_config():
    return {
        'data_pipeline': {
            'fit_tokenizer': True,
            'fit_vectorizer': True,
        },
        'modelling_pipeline': {
            'hyperparameter_tune': True,
        },
        'mlflow': {
            'experiment_name': 'sample_experiment',
            'tracking_uri': 'http://localhost:5000',
        },
    }

def test_main():
    # Mock the load_config function
    with patch('src.pipelines.main.load_config', return_value=sample_config()):
        # Mock the data_pipeline function
        with patch('src.pipelines.main.data_pipeline') as mock_data_pipeline:
            # Mock the modelling_pipeline function
            with patch('src.pipelines.main.modelling_pipeline') as mock_modelling_pipeline:
                # Mock the logger.info function
                with patch('logs.logger.info') as mock_logger_info:
                    # Call the main function
                    main(None)

    # Check if the load_config function was called
    assert load_config.called

    # Check if the data_pipeline function was called
    assert mock_data_pipeline.called

    # Check if the modelling_pipeline function was called
    assert mock_modelling_pipeline.called

    # Check if the logger.info function was called
    assert mock_logger_info.called
