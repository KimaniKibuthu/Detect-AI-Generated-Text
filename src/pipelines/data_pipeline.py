import pickle
from typing import Tuple
from src.data_handling.gather_data import load_data, save_data
from src.data_handling.preprocess import remove_duplicates
from src.data_handling.splitting import split_data
from src.data_handling.validation import validate_schema, DataSchema
from src.data_handling.build_features import train_tokenizer, train_vectorizer, save_vectorizer, tokenize_data, vectorize_data
from logs import logger
from src.utils import load_config


# Load configuration
config = load_config()

def data_pipeline(fit_tokenizer: bool = False, fit_vectorizer: bool = True) -> Tuple:
    """
    Execute the data processing pipeline.

    Parameters:
    - fit_tokenizer (bool): Whether to fit the tokenizer.
    - fit_vectorizer (bool): Whether to fit the vectorizer.

    Returns:
    - Tuple: A tuple containing vectorized X train data, X test, X validation, y train data, y test, and y validation.
    """
    try:
        # Gather data
        logger.info('Gathering data...')
        raw_train_data = load_data(config['data']['preprocessed_train_data_path'])
        raw_test_data = load_data(config['data']['preprocessed_test_data_path'])
        logger.info('Data loaded')
        # Preprocess data
        logger.info('Data preprocessing...')
        processed_train_data = remove_duplicates(raw_train_data, config['data']['preprocessed_train_data_path'])
        processed_test_data = remove_duplicates(raw_test_data, config['data']['preprocessed_test_data_path'])
        logger.info('Data preprocessed successfully')
        # Validate data
        logger.info('Validating data...')
        validate_schema(processed_train_data)
        validate_schema(processed_test_data)
        logger.info('Data validated successfully')
        # Tokenize and featurize data
        ## Tokenize
        logger.info('Tokenizing data...')
        if fit_tokenizer:
            train_tokenizer()

        tokenized_train_data = tokenize_data(processed_train_data, config['models']['sentpiece_model_path'])
        tokenized_test_data = tokenize_data(processed_test_data, config['models']['sentpiece_model_path'])
        logger.info('Data tokenized successfully')
        
        # Split data
        logger.info('Splitting data...')
        validation, test = split_data(tokenized_test_data,
                                      test_size=config['variables']['test_size'])

        logger.info('Data split successfully')
        ## Vectorize
        logger.info('Vectorizing data...')
        if fit_vectorizer:
            vectorizer = train_vectorizer()
            save_vectorizer(vectorizer)
        else:
            with open(config['models']['vectorizer_path'], 'rb') as f:
                vectorizer = pickle.load(f)

        vectorized_X_train_data = vectorize_data(vectorizer, tokenized_train_data['text_spm'])
        vectorized_X_test_data = vectorize_data(vectorizer, test['text_spm'])
        vectorized_X_validation_data = vectorize_data(vectorizer, validation['text_spm'])
        y_train_data = tokenized_train_data['generated'].values
        y_validation_data = validation['generated'].values
        y_test_data = test['generated'].values
        logger.info('Data vectorized successfully')
        
        ## Save the data
        
        save_data(vectorized_X_train_data, config['data']['modelling_X_train_data_path'])
        save_data(vectorized_X_test_data, config['data']['modelling_X_test_data_path'])
        save_data(vectorized_X_validation_data, config['data']['modelling_X_validation_data_path'])
        save_data(y_train_data, config['data']['modelling_y_train_data_path'])
        save_data(y_test_data, config['data']['modelling_y_test_data_path'])
        save_data(y_validation_data, config['data']['modelling_y_validation_data_path'])
        logger.info('Data split and saved successfully')
        logger.info("Data pipeline executed successfully.")

        return vectorized_X_train_data, vectorized_X_test_data, vectorized_X_validation_data, y_train_data, y_test_data, y_validation_data

    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise Exception(f"Error in data pipeline: {str(e)}")


if __name__ == "__main__":
    data_pipeline(config['data_pipeline']['fit_tokenizer'],
                  config['data_pipeline']['fit_vectorizer'])