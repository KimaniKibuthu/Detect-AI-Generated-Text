"""
This module preprocesses the data and returns the processed data.
"""
import pandas as pd
from logs import logger
from src.utils import load_config

config = load_config()

def remove_duplicates(data: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Remove duplicates from the data.

    Parameters:
    - data (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Data without duplicates.
    """
    # Remove duplicates
    n_duplicates = data.duplicated().sum()
    new_data = data.drop_duplicates()
    new_data.reset_index(drop=True, inplace=True)
    logger.info(f"Removed {n_duplicates} duplicates from the data.")
    
    return new_data