import pytest
from unittest.mock import Mock, patch
from src.modelling.train import train_model
from src.modelling.hyperparameters_tuning import objective, hyperparameter_tuning, load_hyperparameters
from src.modelling.evaluate import custom_metric
from src.pipelines.modelling_pipeline import modelling_pipeline, push_models_to_production, get_best_run
from src.utils import load_config, save_config
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import optuna

# Assuming your modules are in src.modelling

# Mock the mlflow module
mlflow = Mock()

# Mock the MlflowClient class
MlflowClient = Mock()

# Define a fixture for sample configuration
@pytest.fixture
def sample_config():
    return {
        'data': {
            'modelling_X_train_data_path': 'path/to/X_train_data.csv',
            'modelling_X_test_data_path': 'path/to/X_test_data.csv',
            'modelling_X_validation_data_path': 'path/to/X_validation_data.csv',
            'modelling_y_train_data_path': 'path/to/y_train_data.csv',
            'modelling_y_test_data_path': 'path/to/y_test_data.csv',
            'modelling_y_validation_data_path': 'path/to/y_validation_data.csv',
        },
        'models': {
            'model_to_use': 'lgbm',
            'hyperparameters_path': 'path/to/hyperparameters.json',
        },
        'variables': {
            'n_trials': 10,
        },
        'modelling_pipeline': {
            'hyperparameter_tune': True,
        }
    }

def test_train_model():
    # Mock the mlflow.start_run() function
    with patch('mlflow.start_run') as mock_start_run:
        # Mock the mlflow.log_params() function
        with patch('mlflow.log_params') as mock_log_params:
            # Mock the mlflow.log_metrics() function
            with patch('mlflow.log_metrics') as mock_log_metrics:
                # Mock the mlflow.xgboost.log_model() function
                with patch('mlflow.xgboost.log_model') as mock_log_model:
                    # Mock the train_model function
                    with patch('src.modelling.train.train_model') as mock_train_model:
                        # Call the train_model function
                        train_model(None, None, None, None)

    # Check if the mlflow.start_run() function was called
    assert mock_start_run.called

    # Check if the mlflow.log_params() function was called
    assert mock_log_params.called

    # Check if the mlflow.log_metrics() function was called
    assert mock_log_metrics.called

    # Check if the mlflow.xgboost.log_model() function was called
    assert mock_log_model.called

    # Check if the train_model function was called
    assert mock_train_model.called

def test_objective():
    # Mock the mlflow.start_run() function
    with patch('mlflow.start_run') as mock_start_run:
        # Mock the mlflow.log_params() function
        with patch('mlflow.log_params') as mock_log_params:
            # Mock the mlflow.log_metrics() function
            with patch('mlflow.log_metrics') as mock_log_metrics:
                # Mock the mlflow.xgboost.log_model() function
                with patch('mlflow.xgboost.log_model') as mock_log_model:
                    # Mock the train_model function
                    with patch('src.modelling.train.train_model') as mock_train_model:
                        # Mock the custom_metric function
                        with patch('src.modelling.evaluate.custom_metric') as mock_custom_metric:
                            # Call the objective function
                            objective(None)

    # Check if the mlflow.start_run() function was called
    assert mock_start_run.called

    # Check if the mlflow.log_params() function was called
    assert mock_log_params.called

    # Check if the mlflow.log_metrics() function was called
    assert mock_log_metrics.called

    # Check if the mlflow.xgboost.log_model() function was called
    assert mock_log_model.called

    # Check if the train_model function was called
    assert mock_train_model.called

    # Check if the custom_metric function was called
    assert mock_custom_metric.called

def test_hyperparameter_tuning():
    # Mock the optuna.create_study() function
    with patch('optuna.create_study') as mock_create_study:
        # Mock the mlflow.start_run() function
        with patch('mlflow.start_run') as mock_start_run:
            # Mock the study.optimize() function
            with patch('study.optimize') as mock_optimize:
                # Call the hyperparameter_tuning function
                hyperparameter_tuning(None, None)

    # Check if the optuna.create_study() function was called
    assert mock_create_study.called

    # Check if the mlflow.start_run() function was called
    assert mock_start_run.called

    # Check if the study.optimize() function was called
    assert mock_optimize.called

def test_load_hyperparameters():
    # Mock the open() function
    with patch('builtins.open', create=True) as mock_open:
        # Mock the json.load() function
        with patch('json.load') as mock_json_load:
            # Call the load_hyperparameters function
            load_hyperparameters()

    # Check if the open() function was called
    assert mock_open.called

    # Check if the json.load() function was called
    assert mock_json_load.called

def test_get_best_run():
    # Mock the MlflowClient class
    with patch('mlflow.tracking.MlflowClient', return_value=MlflowClient):
        # Mock the MlflowClient.search_runs() function
        with patch('MlflowClient.search_runs') as mock_search_runs:
            # Call the get_best_run function
            get_best_run()

    # Check if the MlflowClient.search_runs() function was called
    assert mock_search_runs.called

def test_modelling_pipeline():
    # Mock the load_data function
    with patch('src.modelling.modelling_pipeline.load_data') as mock_load_data:
        # Mock the train_and_evaluate_model function
        with patch('src.modelling.modelling_pipeline.train_and_evaluate_model') as mock_train_and_evaluate_model:
            # Mock the push_models_to_production function
            with patch('src.modelling.modelling_pipeline.push_models_to_production') as mock_push_models_to_production:
                # Call the modelling_pipeline function
                modelling_pipeline(None, True)

    # Check if the load_data function was called
    assert mock_load_data.called

    # Check if the train_and_evaluate_model function was called
    assert mock_train_and_evaluate_model.called

    # Check if the push_models_to_production function was called
    assert mock_push_models_to_production.called
