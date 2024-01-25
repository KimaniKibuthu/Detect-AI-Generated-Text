from typing import Optional, Union, Any
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from logs import logger
from utils import load_config
from sklearn.metrics import make_scorer, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from evaluate import custom_metric, custom_metric_for_lgbm, custom_scorer
from typing import Dict

config = load_config()

def objective(trial: Any) -> float:
    """
    Objective function for hyperparameter tuning of LightGBM model.

    Parameters:
    - trial (Any): Optuna trial object.

    Returns:
    - float: Mean score from cross-validation.
    """
    params: Dict[str, Any] = {
        'objective': config['hyperparameter_tuning_lgbm']['objective'],
        'metric': config['hyperparameter_tuning_lgbm']['metric'],
        'n_iter': config['hyperparameter_tuning_lgbm']['n_iter'],
        'verbose': config['hyperparameter_tuning_lgbm']['verbose'],
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                                              config['hyperparameter_tuning_lgbm']['min_data_in_leaf'][0],
                                              config['hyperparameter_tuning_lgbm']['min_data_in_leaf'][1]),
        'max_depth': trial.suggest_int('max_depth',
                                       config['hyperparameter_tuning_lgbm']['max_depth'][0],
                                       config['hyperparameter_tuning_lgbm']['max_depth'][1]),
        'max_bin': trial.suggest_int('max_bin',
                                     config['hyperparameter_tuning_lgbm']['max_bin'][0],
                                     config['hyperparameter_tuning_lgbm']['max_bin'][1]),
        'learning_rate': trial.suggest_float('learning_rate',
                                             config['hyperparameter_tuning_lgbm']['learning_rate'][0],
                                             config['hyperparameter_tuning_lgbm']['learning_rate'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                config['hyperparameter_tuning_lgbm']['colsample_bytree'][0],
                                                config['hyperparameter_tuning_lgbm']['colsample_bytree'][1]),
        'colsample_bynode': trial.suggest_float('colsample_bynode',
                                                config['hyperparameter_tuning_lgbm']['colsample_bynode'][0],
                                                config['hyperparameter_tuning_lgbm']['colsample_bynode'][1]),
        'lambda_l1': trial.suggest_float('lambda_l1',
                                         config['hyperparameter_tuning_lgbm']['lambda_l1'][0],
                                         config['hyperparameter_tuning_lgbm']['lambda_l1'][1]),
        'lambda_l2': trial.suggest_float('lambda_l2',
                                         config['hyperparameter_tuning_lgbm']['lambda_l2'][0],
                                         config['hyperparameter_tuning_lgbm']['lambda_l2'][1]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LGBMClassifier(**params)

    scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv, scoring=custom_scorer, n_jobs=-1)

    mean_score = np.mean(scores)

    return mean_score
