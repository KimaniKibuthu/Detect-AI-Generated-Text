"""
This module contains code to train the model
"""

from typing import Optional, Union, Any, Dict
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from logs import logger


def train_model(X_train: Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]] = None,
                y_train: Optional[np.ndarray] = None,
                model: Optional[Union[lgb.Booster, xgb.Booster]] = None,
                params: Dict[str, Any] = {}) -> BaseEstimator:
    """
    Train the model.

    Parameters:
    - X_train (Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]]): The training data.
    - y_train (Optional[np.ndarray]): The training labels.
    - params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
    - BaseEstimator: The trained model.
    """
    
    logger.info("Model training started.")
    model.set_params(**params)
    try:    
        logger.info("Training model")
        model.fit(X_train, y_train)
        logger.info("Model trained")
        return model
    except Exception as e:
        logger.error(f"Error training the model: {str(e)}")
        raise Exception(f"Error training the model: {str(e)}")
    

    