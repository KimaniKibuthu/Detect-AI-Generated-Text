"""
Module containing functions for splitting data into train and test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from logs import logger
from gather_data import save_data
from src.utils import load_config

config = load_config()

def split_data(data: pd.DataFrame, test_size: float,  save_path: str) -> pd.DataFrame:
    """
    Split data into train and test sets.

    Parameters:
    - data (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Data without duplicates.
    """
    # Split data
    train, test = train_test_split(data, 
                                   test_size=test_size, 
                                   random_state=config['variables']['random_state'])
    logger.info(f"Split data into train and test sets.")

    # Save data to disk
    save_data(train, save_path + "train.csv")
    save_data(test, save_path + "test.csv")
    logger.info(f"Saved data to '{save_path}'.")

    return train, test