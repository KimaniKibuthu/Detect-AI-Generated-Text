import pytest
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.modelling.train import train_model


# Define a fixture for sample data
@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([0, 1, 0, 1, 0])
    return X_train, y_train

@pytest.mark.parametrize("model", [LGBMClassifier(), XGBClassifier()])
def test_train_model(sample_data, model):
    X_train, y_train = sample_data

    # Test training model with invalid parameters
    invalid_params = {'invalid_param': 'invalid_value'}
    with pytest.raises(Exception):
        train_model(X_train, y_train, model, params=invalid_params)