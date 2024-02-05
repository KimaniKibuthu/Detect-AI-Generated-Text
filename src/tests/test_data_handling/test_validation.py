import pytest
import pandas as pd
from pydantic import ValidationError
from src.utils import load_config
from src.data_handling.validation import DataSchema, validate_schema

# Define a fixture for sample data
@pytest.fixture
def sample_data():
    data = pd.DataFrame({'text': ['sample text 1', 'sample text 2'],
                         'generated': [0, 1]})
    return data

def test_DataSchema_validations():
    # Test valid data
    valid_data = {'text': 'sample text', 'generated': 1}
    assert DataSchema(**valid_data)

    # Test invalid 'text' type
    invalid_text_type = {'text': 123, 'generated': 0}
    with pytest.raises(ValidationError):
        DataSchema(**invalid_text_type)

    # Test invalid 'generated' value
    invalid_generated_value = {'text': 'sample text', 'generated': 2}
    with pytest.raises(ValidationError):
        DataSchema(**invalid_generated_value)

def test_validate_schema(sample_data):
    # Test validating schema with valid data
    errors_valid = validate_schema(sample_data)
    assert not errors_valid

    # Test validating schema with invalid data
    invalid_data = pd.DataFrame({'text': [123, 'sample text 2'],
                                  'generated': [0, 2]})
    errors_invalid = validate_schema(invalid_data)
    assert len(errors_invalid) == 2  # Two validation errors expected

    # Test validating empty DataFrame
    empty_data = pd.DataFrame()
    errors_empty = validate_schema(empty_data)
    assert not errors_empty  
