"""
This module preprocesses the data and returns the processed data.
"""
import pandas as pd
from logs import logger
from utils import load_config
from gather_data import save_data

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
    data = data.drop_duplicates()
    logger.info(f"Removed {n_duplicates} duplicates from the data.")
    
    # Save data to disk
    save_data(data, save_path)

    return data