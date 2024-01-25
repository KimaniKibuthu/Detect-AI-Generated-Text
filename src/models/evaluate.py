"""
This module contains code to evaluate the model
"""

from typing import Optional, Union, Tuple
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from logs import logger
from sklearn.metrics import make_scorer, roc_auc_score, recall_score

def custom_metric(y_true: Optional[np.ndarray] = None,
                  y_pred: Optional[np.ndarray] = None) -> float:
    """
    Custom metric to evaluate the model.

    Parameters:
    - y_true (np.ndarray): True labels
    - y_pred (np.ndarray): Predicted probabilities

    Returns:
    - float: Custom metric value
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    recall_class_1 = recall_score(y_true, (y_pred > 0.5).astype(int), pos_label=1)

    # You can adjust the weights based on your preference
    custom_metric_value = 0.5 * roc_auc + 0.5 * recall_class_1

    return custom_metric_value

def custom_metric_for_lgbm(y_true, y_pred):
    """
    Custom metric function for LightGBM.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted probabilities

    Returns:
    - tuple: Custom metric name, custom metric value, is_higher_better
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    recall_class_1 = recall_score(y_true, (y_pred > 0.5).astype(int), pos_label=1)
    custom_metric_value = 0.5 * roc_auc + 0.5 * recall_class_1
    return 'custom', custom_metric_value, True

def evaluate_model(model: Union[BaseEstimator, lgb.Booster, xgb.Booster],
                   X_test: Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]] = None,
                   y_test: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Evaluate the model.

    Parameters:
    - model (Union[BaseEstimator, lgb.Booster, xgb.Booster]): The model to evaluate
    - X_test (Optional[Union[pd.DataFrame, np.ndarray, csr_matrix]]): The test data
    - y_test (Optional[np.ndarray]): The test labels

    Returns:
    - Tuple[float, float]: A tuple containing ROC-AUC and custom metric values
    """
    logger.info("Model evaluation started.")
    try:
        logger.info("Evaluating model")
        predictions = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, predictions)
        custom = custom_metric(y_test, predictions)
        logger.info("Model evaluated")
        return roc, custom
    except Exception as e:
        logger.error(f"Error evaluating the model: {str(e)}")
        raise Exception(f"Error evaluating the model: {str(e)}")

custom_scorer = make_scorer(custom_metric, needs_proba=True)
