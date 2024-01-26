from typing import Dict, Any
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
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
    mlflow.set_tag("model", "lightgbm-classifier")
    mlflow.log_artifact("data/vectorizer.pkl")
    mlflow.log_artifact("data/sentpiece_model.model")
    mlflow.log_params(params)
        
    model = LGBMClassifier(**params)
    model.fit(X_train_main, y_train_main)
    predictions = model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, predictions)
    custom = custom_metric(y_val, predictions)
    mlflow.log_metric("roc", roc)
    mlflow.log_metric("roc and recall", custom)
    # Log the model
    artifact_path = "model"
    signature = infer_signature(X_train_main, model.predict(X_train_main))
    mlflow.lightgbm.log_model(model, artifact_path, signature=signature)
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
