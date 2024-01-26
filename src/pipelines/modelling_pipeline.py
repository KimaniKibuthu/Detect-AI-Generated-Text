import json
import optuna
from data_handling.gather_data import load_data
from modelling.evaluate import evaluate
from modelling.hyperparameters_tuning import objective
from modelling.train import train_model
from logs import logger
import mlflow
from utils import load_config

config = load_config()

mlflow.set_experiment(config['mlflow']['experiment_name'])
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
PRODUCTION_TAG = 'production'


def get_production_model_roc():
    """
    Fetch and return ROC of the model currently in production.

    Returns:
    - float: ROC of the model in production or 0 if no model is in production.
    """
    production_runs = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(config['mlflow']['experiment_name']).experiment_id],
        filter_string=f"tags.mlflow.runName = '{PRODUCTION_TAG}'"
    )
    if not production_runs.empty:
        return production_runs.iloc[0]['metrics.roc']
    return 0.0


def train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune):
    """
    Train and evaluate the model.

    Parameters:
    - X_train, y_train, X_validation, y_validation: Training and validation data and labels.
    - hyperparameter_tune (bool): Whether to tune the hyperparameters.

    Returns:
    - model: Trained model.
    - roc: ROC score on the validation set.
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


def modelling_pipeline(hyperparameter_tune: bool = False) -> None:
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
    production_roc = get_production_model_roc()

    if production_roc > 0:
        # Model is in production, evaluate new model on validation set
        logger.info('Model in production. Evaluating new model...')
        new_model, new_roc = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)

        # Check if new model has better ROC than the model in production
        if new_roc > production_roc:
            # New model has better performance, move it to production
            logger.info('New model has better performance. Pushing to production...')
            mlflow.set_tag('mlflow.runName', PRODUCTION_TAG)
            mlflow.lightgbm.log_model(new_model, "model")
        else:
            logger.info('New model does not have better performance than the model in production.')

    else:
        # No model in production, train and push the model to production
        logger.info('No model in production. Training and pushing the model to production...')
        new_model, _ = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)

        # Push the model to production
        logger.info('Pushing the model to production...')
        mlflow.set_tag('mlflow.runName', PRODUCTION_TAG)
        mlflow.lightgbm.log_model(new_model, "model")

    logger.info('Modelling pipeline finished')
