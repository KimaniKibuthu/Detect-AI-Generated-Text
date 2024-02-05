import pytest
import pandas as pd
from src.pipelines.data_pipeline import data_pipeline
from src.utils import load_config

# Assuming your module is named 'src.data_processing.data_pipeline'

# Define a fixture for sample configuration
@pytest.fixture
def sample_config():
    return {
        'data': {
            'preprocessed_train_data_path': 'path/to/preprocessed_train_data.csv',
            'preprocessed_test_data_path': 'path/to/preprocessed_test_data.csv',
            'modelling_X_train_data_path': 'path/to/X_train_data.csv',
            'modelling_X_test_data_path': 'path/to/X_test_data.csv',
            'modelling_X_validation_data_path': 'path/to/X_validation_data.csv',
            'modelling_y_train_data_path': 'path/to/y_train_data.csv',
            'modelling_y_test_data_path': 'path/to/y_test_data.csv',
            'modelling_y_validation_data_path': 'path/to/y_validation_data.csv',
        },
        'models': {
            'sentpiece_model_path': 'path/to/sentpiece_model.pkl',
            'vectorizer_path': 'path/to/vectorizer.pkl',
        },
        'variables': {
            'test_size': 0.2,
        },
        'data_pipeline': {
            'fit_tokenizer': True,
            'fit_vectorizer': True,
        }
    }

def test_data_pipeline(sample_config):
    # Test data pipeline with sample configuration
    config = sample_config
    result = data_pipeline(config['data_pipeline']['fit_tokenizer'], config['data_pipeline']['fit_vectorizer'])

    # Check if the result is a tuple
    assert isinstance(result, tuple)

    # Check if the tuple contains the expected elements
    assert len(result) == 6
    assert all(isinstance(element, pd.DataFrame) or isinstance(element, pd.Series) for element in result)
