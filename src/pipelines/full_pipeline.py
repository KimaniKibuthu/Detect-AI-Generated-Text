from src.pipelines.data_pipeline import data_pipeline
from src.pipelines.modelling_pipeline import modelling_pipeline
from src.utils import load_config
from logs import logger

def main(config):
    """
    Execute the full pipeline.
    """
    # Data pipeline
    logger.info('Executing full pipeline...')
    data_pipeline(config['data_pipeline']['fit_tokenizer'],
                  config['data_pipeline']['fit_vectorizer'])
    # Modelling pipeline
    modelling_pipeline(config['modelling_pipeline']['hyperparameter_tune'])
    logger.info('Full pipeline finished')

if __name__ == '__main__':
    config = load_config()
    main(config)