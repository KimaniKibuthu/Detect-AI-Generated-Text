import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
from src.utils import load_config
from src.modelling.evaluate import custom_metric, evaluate_model, custom_metric_for_lgbm, custom_scorer


# Define a fixture for sample data
@pytest.fixture
def sample_data():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                      'feature2': [5, 4, 3, 2, 1]})
    y = np.array([0, 1, 0, 1, 0])
    return X, y

def test_custom_metric(sample_data):
    _, y_true = sample_data
    y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.4])

    # Test custom metric function
    custom_metric_value = custom_metric(y_true, y_pred)
    assert isinstance(custom_metric_value, float)

def test_custom_metric_for_lgbm(sample_data):
    _, y_true = sample_data
    y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.4])

    # Test custom metric function for LightGBM
    metric_name, metric_value, is_higher_better = custom_metric_for_lgbm(y_true, y_pred)
    assert isinstance(metric_name, str)
    assert isinstance(metric_value, float)
    assert isinstance(is_higher_better, bool)

def test_evaluate_model(sample_data):
    X_test, y_test = sample_data

    # Test evaluating model with a simple estimator
    class DummyEstimator(BaseEstimator):
        def fit(self):
            pass
        def predict_proba(self):
            return np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])

    dummy_model = DummyEstimator()
    roc, custom = evaluate_model(dummy_model, X_test, y_test)
    
    # Check if the results are floats
    assert isinstance(roc, float)
    assert isinstance(custom, float)

def test_custom_scorer(sample_data):
    X_test, y_test = sample_data

    # Test custom scorer with a simple estimator
    class DummyEstimator(BaseEstimator):
        def fit(self):
            pass
        def predict_proba(self):
            return np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])

    dummy_model = DummyEstimator()
    score = custom_scorer(dummy_model, X_test, y_test)

    # Check if the result is a float
    assert isinstance(score, float)
