import json
import optuna
from sklearn.metrics import roc_auc_score
from src.data_handling.gather_data import load_data
from src.modelling.evaluate import custom_metric
from src.modelling.hyperparameters_tuning import objective
from src.modelling.train import train_model
from logs import logger
import mlflow
from src.utils import load_config
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    if not production_runs.empty and 'roc' in production_runs.columns:
        # Assuming 'roc' is the correct column name
        return production_runs.iloc[0]['roc']
    else:
        print("No data or 'roc' column not found in the DataFrame.")
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
    model_to_use = config['models']['model_to_use']
    model = LGBMClassifier() if model_to_use == 'lgbm' else XGBClassifier()

    if hyperparameter_tune:
        study = optuna.create_study(direction='maximize')
        with mlflow.start_run():
            if model_to_use == 'lgbm':
                mlflow.lightgbm.autolog()
            else:
                mlflow.xgboost.autolog()

            study.optimize(objective, n_trials=config['variables']['n_trials'])
            best_params = study.best_params
            logger.info("Best Hyperparameters:", best_params)
            mlflow.log_params(best_params)

        with open(config['models']['hyperparameters_path'], 'w') as json_file:
            json.dump(best_params, json_file)

        model = train_model(X_train, y_train, model, best_params)

    with open(config['models']['hyperparameters_path'], 'r') as json_file:
        params = json.load(json_file)
    
    with mlflow.start_run():    	   
        model = train_model(X_train, y_train, model, params)

        logger.info('Model trained')

        # Evaluate the model on the validation set
        logger.info('Evaluating model on the validation set...')
        predictions = model.predict_proba(X_validation)[:, 1]
        roc = roc_auc_score(y_validation, predictions)
        custom_metric_value = custom_metric(y_validation, predictions)

        mlflow.log_metric("roc", roc)
        mlflow.log_metric("custom_metric", custom_metric_value)
        if model_to_use == 'lgbm':
            mlflow.lightgbm.log_model(model, 'model')
        else:
            mlflow.xgboost.log_model(model, 'model')

    logger.info(f'The ROC on the validation set is {roc} and the custom metric is {custom_metric_value}.')

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
            mlflow.sklearn.log_model(new_model, "model")
        else:
            logger.info('New model does not have better performance than the model in production.')

    else:
        # No model in production, train and push the model to production
        logger.info('No model in production. Training and pushing the model to production...')
        new_model, _ = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)

        # Push the model to production
        logger.info('Pushing the model to production...')
        mlflow.set_tag('mlflow.runName', PRODUCTION_TAG)
        mlflow.sklearn.log_model(new_model, "model")

    logger.info('Modelling pipeline finished')


if __name__ == '__main__':
    modelling_pipeline(config['modelling_pipeline']['hyperparameter_tune'])
