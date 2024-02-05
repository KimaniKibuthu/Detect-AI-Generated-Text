import pytest
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_handling.gather_data import load_data
from src.utils import load_config
import scipy
from src.data_handling.build_features import (
    train_tokenizer,
    train_vectorizer,
    save_vectorizer,
    tokenize_data,
    vectorize_data
)



@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def mock_train_data(tmpdir, config):
    # Create a temporary directory for the mock training data
    train_data_path = tmpdir.join("train_data.csv")
    mock_data = pd.DataFrame({"text": ["This is a sample text.", "Another example."]})
    mock_data.to_csv(train_data_path, index=False)
    config['data']['preprocessed_train_data_path'] = train_data_path
    return config

def test_train_tokenizer(mock_train_data):
    train_tokenizer()
    assert os.path.exists(mock_train_data['models']['sentpiece_model_prefix'] + ".model")

def test_train_vectorizer(mock_train_data):
    vectorizer = train_vectorizer()
    assert isinstance(vectorizer, TfidfVectorizer)

def test_save_vectorizer(mock_train_data):
    vectorizer = TfidfVectorizer()
    save_vectorizer(vectorizer)
    assert os.path.exists(mock_train_data['models']['vectorizer_path'])

def test_tokenize_data(mock_train_data):
    train_df = load_data(mock_train_data['data']['preprocessed_train_data_path'])
    tok_model_path = mock_train_data['models']['sentpiece_model_prefix'] + ".model"
    tokenized_data = tokenize_data(train_df, tok_model_path)
    assert 'tokens' in tokenized_data.columns
    assert 'text_spm' in tokenized_data.columns

def test_vectorize_data(mock_train_data):
    vectorizer = train_vectorizer()
    train_df = load_data(mock_train_data['data']['preprocessed_train_data_path'])
    vectorized_data = vectorize_data(vectorizer, train_df['text'])
    assert isinstance(vectorized_data, scipy.sparse.csr_matrix)
    assert vectorized_data.shape[0] == len(train_df)
