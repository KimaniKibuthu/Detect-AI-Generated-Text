import json
import optuna
from sklearn.metrics import roc_auc_score
from src.data_handling.gather_data import load_data
from src.modelling.evaluate import custom_metric
from src.modelling.hyperparameters_tuning import objective
from src.modelling.train import train_model
from logs import logger
import mlflow
from mlflow.tracking import MlflowClient
from src.utils import load_config, save_config
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import json

# Constants
EXPERIMENT_NAME_KEY = 'mlflow'
METRIC_TO_USE = 'roc'
MODEL_NAME = 'detect-ai-text-model'

config = load_config()
mlflow.set_experiment(config[EXPERIMENT_NAME_KEY]['experiment_name'])
mlflow.set_tracking_uri(config[EXPERIMENT_NAME_KEY]['tracking_uri'])
client = MlflowClient()

def push_models_to_production(client):
    """
    Compare the ROC of the model currently in production with the ROC of the best model.

    Returns:
    - None
    """
    # Get the best model in a run
    best_run, best_run_metric_value = get_best_run()

    if not best_run:
        logger.info("No runs found. Exiting...")
        return

    best_run_id = best_run.info.run_id

    # Register the model
    result = mlflow.register_model(f"runs:/{best_run_id}/model",
                                   MODEL_NAME,
                                   tags={METRIC_TO_USE: best_run_metric_value})
    
    config['models']['model_uri'] = f"runs:/{best_run_id}/model"
    
    save_config(config)
    # Check for the best roc-recall results and push that model to production
    model_version_details = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    
    if model_version_details:
        
        production_roc_recall = float(model_version_details[0].tags.get(METRIC_TO_USE, 0.0))
        if production_roc_recall < best_run_metric_value:
            archive_current_model(model_version_details[0].version)
            move_model_to_production(result.version)

    


def get_best_run():
    """
    Get the best model run based on ROC-recall.

    Returns:
    - Tuple(mlflow.entities.Run, float): Best model run and corresponding metric value.
    """
    best_run = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(config[EXPERIMENT_NAME_KEY]['experiment_name']).experiment_id],
        order_by=[f"metrics.{METRIC_TO_USE} DESC"],
        max_results=1
    )[0]
    
    
    best_run_metric_value = float(best_run.data.metrics.get(METRIC_TO_USE, 0.0))
    
    return best_run, best_run_metric_value


def archive_current_model(version):
    """
    Archive the model with the specified version.

    Parameters:
    - version: Model version to archive.

    Returns:
    - None
    """
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Archived"
    )


def move_model_to_production(version):
    """
    Move the model with the specified version to production.

    Parameters:
    - version: Model version to move to production.

    Returns:
    - None
    """
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production"
    )


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
        best_params = hyperparameter_tuning(X_train, y_train)
        model = train_model(X_train, y_train, model, best_params)
    else:
        best_params = load_hyperparameters()
        model = train_model(X_train, y_train, model, best_params)
    logger.info('Model trained')

    with mlflow.start_run():
        evaluate_and_log_metrics(model, X_validation, y_validation)
        log_model(model, model_to_use)
        roc = roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1])

    return model, roc



def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning and return the best parameters.

    Parameters:
    - X_train, y_train: Training data and labels.

    Returns:
    - dict: Best hyperparameters.
    """
    study = optuna.create_study(direction='maximize')
    with mlflow.start_run():
        study.optimize(objective, n_trials=config['variables']['n_trials'])
        best_params = study.best_params
        logger.info("Best Hyperparameters:", best_params)
        mlflow.log_params(best_params)
        mlflow.log_metrics(METRIC_TO_USE, study.best_value)

    with open(config['models']['hyperparameters_path'], 'w') as json_file:
        json.dump(best_params, json_file)

    return best_params


def load_hyperparameters():
    """
    Load hyperparameters from a file.

    Returns:
    - dict: Loaded hyperparameters.
    """
    with open(config['models']['hyperparameters_path'], 'r') as json_file:
        return json.load(json_file)


def evaluate_and_log_metrics(model, X_validation, y_validation):
    """
    Evaluate the model on the validation set and log metrics.

    Parameters:
    - model: Trained model.
    - X_validation, y_validation: Validation data and labels.

    Returns:
    - None
    """
    logger.info('Evaluating model on the validation set...')
    predictions = model.predict_proba(X_validation)[:, 1]
    roc = roc_auc_score(y_validation, predictions)
    custom_metric_value = custom_metric(y_validation, predictions)

    mlflow.log_metric("roc", roc)
    mlflow.log_metric("roc_recall", custom_metric_value)

    logger.info(f'The ROC on the validation set is {roc} and the custom metric is {custom_metric_value}.')


def log_model(model, model_to_use):
    """
    Log the trained model.

    Parameters:
    - model: Trained model.
    - model_to_use: Model type ('lgbm' or 'xgboost').

    Returns:
    - None
    """
    if model_to_use == 'lgbm':
        mlflow.lightgbm.log_model(model, 'model')
    else:
        mlflow.xgboost.log_model(model, 'model')


def modelling_pipeline(client, hyperparameter_tune: bool = False) -> None:
    """
    Execute the modelling pipeline.

    Parameters:
    - client: MlflowClient instance.
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
    

    logger.info('Model Pipeline started...')
    model, _ = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)
    push_models_to_production(client)
    logger.info('Model Pipeline completed')
    # else:
    #     logger.info('Training model...')
    #     model, _ = train_and_evaluate_model(X_train, y_train, X_validation, y_validation, hyperparameter_tune)
    #     push_models_to_production(client)
    #     logger.info('Model trained')


if __name__ == '__main__':
    modelling_pipeline(client, config['modelling_pipeline']['hyperparameter_tune'])
