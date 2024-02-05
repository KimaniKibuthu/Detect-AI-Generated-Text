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

def test_train_model(sample_data):
    X_train, y_train = sample_data

    # Test training model with LGBMClassifier
    lgbm_model = LGBMClassifier()
    trained_lgbm_model = train_model(X_train, y_train, lgbm_model)

    # Check if the result is a trained LGBM model
    assert isinstance(trained_lgbm_model, LGBMClassifier)

    # Test training model with XGBClassifier
    xgb_model = XGBClassifier()
    trained_xgb_model = train_model(X_train, y_train, xgb_model)

    # Check if the result is a trained XGB model
    assert isinstance(trained_xgb_model, XGBClassifier)

    # Test training model with invalid parameters
    invalid_params = {'invalid_param': 'invalid_value'}
    with pytest.raises(Exception):
        train_model(X_train, y_train, lgbm_model, params=invalid_params)
        train_model(X_train, y_train, xgb_model, params=invalid_params)
