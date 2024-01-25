from data_pipeline import data_pipeline
from modelling_pipeline import model_pipeline, train_and_evaluate_model
from logs import logger
def main():
    """
    Execute the full pipeline.
    """
    # Data pipeline
    logger.info('Executing full pipeline...')
    data_pipeline()
    # Modelling pipeline
    model_pipeline()
    logger.info('Full pipeline finished')

if __name__ == '__main__':
    main()