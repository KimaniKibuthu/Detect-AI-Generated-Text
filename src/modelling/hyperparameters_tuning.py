from typing import Dict, Any
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils import load_config
from src.modelling.evaluate import custom_scorer

config = load_config()

def objective(trial: Any) -> float:
    """
    Objective function for hyperparameter tuning of LightGBM model.

    Parameters:
    - trial (Any): Optuna trial object.

    Returns:
    - float: Mean score from cross-validation.
    """
    global X_train, y_train
    params: Dict[str, Any] = {
        'objective': config['hyperparameter_tuning_lgbm']['objective'],
        'metric': config['hyperparameter_tuning_lgbm']['metric'],
        'n_iter': config['hyperparameter_tuning_lgbm']['n_iter'],
        'verbose': config['hyperparameter_tuning_lgbm']['verbose'],
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                                              *config['hyperparameter_tuning_lgbm']['min_data_in_leaf']),
        'max_depth': trial.suggest_int('max_depth',
                                       *config['hyperparameter_tuning_lgbm']['max_depth']),
        'max_bin': trial.suggest_int('max_bin',
                                     *config['hyperparameter_tuning_lgbm']['max_bin']),
        'learning_rate': trial.suggest_float('learning_rate',
                                             *config['hyperparameter_tuning_lgbm']['learning_rate']),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                *config['hyperparameter_tuning_lgbm']['colsample_bytree']),
        'colsample_bynode': trial.suggest_float('colsample_bynode',
                                                *config['hyperparameter_tuning_lgbm']['colsample_bynode']),
        'lambda_l1': trial.suggest_float('lambda_l1',
                                         *config['hyperparameter_tuning_lgbm']['lambda_l1']),
        'lambda_l2': trial.suggest_float('lambda_l2',
                                         *config['hyperparameter_tuning_lgbm']['lambda_l2']),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_to_use = config['models']['model_to_use']
    if model_to_use == 'lgbm':
        model = LGBMClassifier(**params)
    else:
        model = XGBClassifier(**params)
    
    scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv, scoring=custom_scorer, n_jobs=-1)

    mean_score = np.mean(scores)

    return mean_score
