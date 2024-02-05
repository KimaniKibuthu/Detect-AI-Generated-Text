import pandas as pd
import pytest
from src.data_handling.preprocess import remove_duplicates


# Define a fixture for sample data
@pytest.fixture
def sample_data():
    data = pd.DataFrame({'col1': [1, 2, 2, 3, 4],
                         'col2': ['a', 'b', 'b', 'c', 'd']})
    return data

def test_remove_duplicates(sample_data):
    # Test removing duplicates
    processed_data = remove_duplicates(sample_data)
    
    # Check if the result is a DataFrame
    assert isinstance(processed_data, pd.DataFrame)
    
    # Check if duplicates are removed
    assert len(processed_data) == len(set(sample_data.drop_duplicates().index))
    
    # Check if the index is reset
    assert processed_data.index.tolist() == list(range(len(processed_data)))
