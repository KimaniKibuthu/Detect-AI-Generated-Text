"""
Module containing functions for splitting data into train and test sets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from logs import logger
from src.utils import load_config
from typing import Optional, Union, Tuple
from scipy.sparse import csr_matrix
import numpy as np

config = load_config()

def split_data(data: Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]],
               test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Parameters:
    - X_data (Union[pd.DataFrame, np.ndarray, csr_matrix]): Input features.
    - y_data (Optional[np.ndarray]): Target variable.
    - test_size (float): Size of the test set.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: Tuple containing X_train, X_test, y_train, y_test.
    """
    # Split data
    train, test= train_test_split(data,
                                  test_size=test_size,
                                  random_state=config['variables']['random_state'])
    logger.info(f"Split data into train and test sets.")

    return train, test
