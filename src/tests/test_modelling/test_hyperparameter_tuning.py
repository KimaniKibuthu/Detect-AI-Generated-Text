import optuna
import pytest
import numpy as np
from src.utils import load_config
from src.modelling.evaluate import custom_scorer
from src.modelling.hyperparameters_tuning import objective

# Assuming your module is named 'your_module'

# Define a fixture for sample data
@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([0, 1, 0, 1, 0])
    return X_train, y_train

def test_objective(sample_data):
    pass

    # Test objective function with Optuna trial
    def dummy_objective(trial):
        return objective(trial)

    study = optuna.create_study(direction='maximize')
    study.optimize(dummy_objective, n_trials=5)

    # Check if the study is not empty
    assert len(study.trials) > 0

    # Check if the best trial has a valid score
    assert isinstance(study.best_value, float)

    # Check if the best trial has valid parameters
    assert isinstance(study.best_params, dict)

    # Check if the best trial's parameters are within the specified range
    assert all(param_name in study.best_params for param_name in ['min_data_in_leaf', 'max_depth', 'max_bin',
                                                                  'learning_rate', 'colsample_bytree',
                                                                  'colsample_bynode', 'lambda_l1', 'lambda_l2'])

    # Check if the best trial's score is a float
    assert isinstance(study.best_value, float)

    # Check if the best trial's parameters are within the specified range
    assert all(isinstance(param_value, (int, float)) for param_value in study.best_params.values())

    # Check if the best trial's score is within a reasonable range
    assert 0 <= study.best_value <= 1
