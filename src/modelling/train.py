"""
This module contains code to train the model
"""

from logs import logger
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any
from scipy.sparse import csr_matrix
import lightgbm as lgb
import xgboost as xgb

def train_model(X_train: Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]] = None,
                y_train: Optional[np.ndarray] = None,
                model: Optional[Union[lgb.LGBMClassifier, xgb.XGBClassifier]] = None,
                params: Dict[str, Any] = {}) -> Union[lgb.LGBMClassifier, xgb.XGBClassifier]:
    """
    Train the model.

    Parameters:
    - X_train (Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]]): The training data.
    - y_train (Optional[np.ndarray]): The training labels.
    - model (Optional[Union[lgb.LGBMClassifier, xgb.XGBClassifier]]): The model to be trained.
    - params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
    - Union[lgb.LGBMClassifier, xgb.XGBClassifier]: The trained model.
    """
    if not isinstance(X_train, (pd.DataFrame, np.ndarray, csr_matrix)):
        raise ValueError("X_train must be a pandas DataFrame, numpy array, or scipy csr_matrix.")
    if not isinstance(y_train, np.ndarray):
        raise ValueError("y_train must be a numpy array.")
    if not isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
        raise ValueError("model must be a LightGBM LGBMClassifier or XGBoost XGBClassifier instance.")
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")
    
    logger.info("Model training started.")
    
    if not any(param in model.get_params() for param in params):
        raise ValueError("Invalid hyperparameters provided.")
    model.set_params(**params)
    try:    
        logger.info("Training model")
        model.fit(X_train, y_train)
        logger.info("Model trained")
        return model
    except Exception as e:
        logger.error(f"Error training the model: {str(e)}")
        raise Exception(f"Error training the model: {str(e)}")

    

    