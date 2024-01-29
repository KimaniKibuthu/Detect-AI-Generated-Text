from src.pipelines.data_pipeline import data_pipeline
from src.pipelines.modelling_pipeline import modelling_pipeline
from src.utils import load_config
from logs import logger
import mlflow
from mlflow.tracking import MlflowClient
# Constants
EXPERIMENT_NAME_KEY = 'mlflow'
METRIC_TO_USE = 'roc'
MODEL_NAME = 'detect-ai-text-model'

config = load_config()
mlflow.set_experiment(config[EXPERIMENT_NAME_KEY]['experiment_name'])
mlflow.set_tracking_uri(config[EXPERIMENT_NAME_KEY]['tracking_uri'])
client = MlflowClient()


def main(config):
    """
    Execute the full pipeline.
    """
    # Data pipeline
    logger.info('Executing full pipeline...')
    data_pipeline(config['data_pipeline']['fit_tokenizer'],
                  config['data_pipeline']['fit_vectorizer'])
    # Modelling pipeline
    modelling_pipeline(client, config['modelling_pipeline']['hyperparameter_tune'])
    logger.info('Full pipeline finished')

if __name__ == '__main__':
    config = load_config()
    main(config)