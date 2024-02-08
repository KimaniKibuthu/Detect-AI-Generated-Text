import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from src.pipelines.data_pipeline import data_pipeline
from src.utils import load_config



# Define a fixture for sample configuration
@pytest.fixture
def sample_config():
    return load_config()


def test_data_pipeline(sample_config):
    # Test data pipeline with sample configuration
    config = sample_config
    result = data_pipeline(config['data_pipeline']['fit_tokenizer'], config['data_pipeline']['fit_vectorizer'])

    # Check if the result is a tuple
    assert isinstance(result, tuple)

    # Check if the tuple contains the expected elements
    assert len(result) == 6
    assert all(isinstance(element, (pd.DataFrame, pd.Series, np.ndarray, csr_matrix)) for element in result)
