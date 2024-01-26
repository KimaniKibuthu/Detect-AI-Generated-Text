"""
Build features for the model
"""
import pickle
import scipy
import sentencepiece as spm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logs import logger
from src.data_handling.gather_data import load_data
from src.utils import load_config

config = load_config()

def train_tokenizer() -> None:
    """
    Train SentencePiece tokenizer.

    Loads the processed training data and trains a SentencePiece tokenizer model.
    The trained model is saved using the specified model prefix and vocabulary size.
    """
    train_sentpiece_df = load_data(config['data']['preprocessed_train_data_path'])
    with open(config['data']['sentpiece_train_data_path'], 'w', encoding='utf-8') as f:
        for text in train_sentpiece_df['text']:
            f.write(text + '\n')

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(input=config['data']['sentpiece_train_data_path'],
                                   model_prefix=config['models']['sentpiece_model_prefix'],
                                   vocab_size=config['variables']['vocab_size'])

def train_vectorizer() -> TfidfVectorizer:
    """
    Train TF-IDF vectorizer.

    Loads the processed training data and trains a TF-IDF vectorizer.
    Returns the fitted vectorizer.

    Returns:
    - TfidfVectorizer: Fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(ngram_range=tuple(config['variables']['ngram_range']),
                                 sublinear_tf=config['variables']['sublinear_tf'],
                                 lowercase=config['variables']['lowercase'],
                                 max_features=config['variables']['max_features'])
    train_df = load_data(config['data']['preprocessed_train_data_path'])
    vectorizer.fit_transform(train_df['text'])
    return vectorizer

def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """
    Save TF-IDF vectorizer to a file.

    Args:
    - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    with open(config['models']['vectorizer_path'], 'wb') as f:
        pickle.dump(vectorizer, f)

def tokenize_data(data: pd.DataFrame, tok_model_path: str) -> pd.DataFrame:
    """
    Tokenize the data using the sentencepiece model.

    Args:
        data (pd.DataFrame): The data to tokenize
        tok_model_path (str): The path to the sentencepiece model

    Returns:
        pd.DataFrame: The tokenized data
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_model_path)
    data.loc[:, 'tokens'] = data['text'].apply(lambda x: sp.EncodeAsPieces(x.lower()))
    data.loc[:, 'text_spm'] = data['tokens'].apply(lambda x: ' '.join(x))
    return data

def vectorize_data(vectorizer: TfidfVectorizer, text_column: pd.Series) -> scipy.sparse.csr_matrix:
    """
    Vectorize text data using a TF-IDF vectorizer.

    Args:
        - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        - text_column (pd.Series): Text data to be vectorized.

    Returns:
        - scipy.sparse.csr_matrix: Vectorized data.
    """
    vectorized_data = vectorizer.transform(text_column)
    return vectorized_data


    
    
    
