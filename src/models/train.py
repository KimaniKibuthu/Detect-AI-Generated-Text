"""
This module contains code to train the model
"""
import mlflow
from typing import Optional, Union, Any
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from logs import logger


def train_model(model: Union[BaseEstimator, lgb.Booster, xgb.Booster],
                X_train: Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]] = None,
                y_train: Optional[np.ndarray] = None) -> None:
    """
    Train the model.

    Parameters:
    - model (model): The model to train
    - X_train (pd.DataFrame): The training data
    - y_train (pd.DataFrame): The training labels

    Returns:
    - sklearn model: The trained model
    """
    
    logger.info("Model training started.")
    try:    
        logger.info("Training model")
        model.fit(X_train, y_train)
        logger.info("Model trained")
        return model
    except Exception as e:
        logger.error(f"Error training the model: {str(e)}")
        raise Exception(f"Error training the model: {str(e)}")
    
    