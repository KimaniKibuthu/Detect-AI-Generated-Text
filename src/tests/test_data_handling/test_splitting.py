import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_handling.splitting import split_data

# Define a fixture for sample data
@pytest.fixture
def sample_data():
    data = pd.DataFrame({'col1': [1, 2, 3, 4, 5],
                         'col2': ['a', 'b', 'c', 'd', 'e'],
                         'target': [0, 1, 0, 1, 0]})
    return data

def test_split_data(sample_data):
    # Test splitting data
    train, test = split_data(sample_data, test_size=0.2)
    
    # Check if the results are DataFrames
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
    # Check if the sum of lengths equals the original length
    assert len(train) + len(test) == len(sample_data)
    
    # Check if the columns are preserved
    assert list(train.columns) == list(test.columns) == list(sample_data.columns)
    
    # Check if the target variable distribution is preserved
    assert train['target'].sum() + test['target'].sum() == sample_data['target'].sum()
