import mlflow
import pickle
import sentencepiece as spm
import pandas as pd
from logs import logger
from src.utils import load_config
from src.data_handling.build_features import tokenize_data, vectorize_data

config = load_config()

def get_inference(data: str) -> tuple:
    """
    Get the predictionload_config
    Parameters:
    - data (str): Input data.
    

    Returns:
    - List: Predictions 
    """
    data_df = pd.DataFrame(columns=['text'], data=[data])
    # Load the vectorizer from pkl file
    vectorizer_path = config['models']['vectorizer_path']
    sentpiece_path = config['models']['sentpiece_model_path']
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Tokenize data
    logger.info('Tokenizing data...')
    tok_data = tokenize_data(data_df, sentpiece_path)
    
    # vectorize data
    logger.info('Vectorizing data...')
    vectorized_data = vectorize_data(vectorizer, tok_data["text_spm"])
    
    # Get predictions from model
    loaded_model_uri = config["models"]["model_uri"]
    
    # Load the model as a PyFuncModel
    loaded_model = mlflow.pyfunc.load_model(loaded_model_uri)
    predictions = loaded_model.predict(vectorized_data)
    


    return predictions
