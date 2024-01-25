"""
This contains the modelling pipeline
"""

import mlflow
import optuna
import json
from data_handling.gather_data import load_data
from models.evaluate import evaluate
from models.hyperparameters_tuning import objective
from models.train import train_model
from logs import logger
from utils import load_config

config = load_config()

mlflow.set_experiment(config['mlflow']['experiment_name'])
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

def train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune):
    """
    Train and evaluate the model.

    Parameters:
    - X_train, y_train, X_validation, y_validation: Training and validation data and labels.
    - hyperparameter_tune (bool): Whether to tune the hyperparameters.

    Returns:
    - model: Trained model.
    """
    logger.info('Training model...')
    if hyperparameter_tune:
        study = optuna.create_study(direction='maximize')
        with mlflow.start_run():
            mlflow.lightgbm.autolog()
            study.optimize(objective, n_trials=config['variables']['n_trials'])
            best_params = study.best_params

        with open(config['models']['hyperparameters_path'], 'w') as json_file:
            json.dump(best_params, json_file)

        logger.info("Best Hyperparameters:", best_params)
        model = train_model(X_train, y_train, best_params)
    else:
        with open(config['models']['hyperparameters_path'], 'r') as json_file:
            params = json.load(json_file)
        model = train_model(X_train, y_train, params)

    logger.info('Model trained')

    # Evaluate the model on the validation set
    logger.info('Evaluating model on the validation set...')
    roc, custom = evaluate(model, X_validation, y_validation)
    logger.info(f'The ROC on the validation set is {roc} and the custom metric is {custom}.')

    return model, roc

def modelling_pipeline(hyperparameter_tune: bool = True) -> None:
    """
    Execute the modelling pipeline.

    Parameters:
    - hyperparameter_tune (bool): Whether to tune the hyperparameters.

    Returns:
    - None
    """
    logger.info('Starting modelling pipeline...')

    # Load data
    logger.info('Loading data...')
    X_train = load_data(config['data']['modelling_X_train_data_path'])
    X_test = load_data(config['data']['modelling_X_test_data_path'])
    X_validation = load_data(config['data']['modelling_X_validation_data_path'])
    y_train = load_data(config['data']['modelling_y_train_data_path'])
    y_test = load_data(config['data']['modelling_y_test_data_path'])
    y_validation = load_data(config['data']['modelling_y_validation_data_path'])
    logger.info('Data loaded')

    # Check if a model is in production
    is_model_in_production = mlflow.search_runs(
        experiment_ids=[config['mlflow']['experiment_name']],
        filter_string="tags.mlflow.runName = 'production'"
    ).shape[0] > 0

    if is_model_in_production:
        # Model is in production, evaluate new model on validation set
        logger.info('Model in production. Evaluating new model...')
        new_model, new_roc = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)

        # Check if new model has better ROC than the model in production
        production_model_run = mlflow.search_runs(
            experiment_ids=[config['mlflow']['production_experiment_id']],
            filter_string="tags.mlflow.runName = 'production'"
        ).iloc[0]
        production_roc = production_model_run['metrics.roc']

        if new_roc > production_roc:
            # New model has better performance, move it to production
            logger.info('New model has better performance. Pushing to production...')
            mlflow.set_tag('mlflow.runName', 'production')
            mlflow.lightgbm.log_model(new_model, "model")
        else:
            logger.info('New model does not have better performance than the model in production.')

    else:
        # No model in production, train and push the model to production
        logger.info('No model in production. Training and pushing the model to production...')
        new_model, _ = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)

        # Push the model to production
        logger.info('Pushing the model to production...')
        mlflow.set_tag('mlflow.runName', 'production')
        mlflow.lightgbm.log_model(new_model, "model")

    logger.info('Modelling pipeline finished')




