import os
import pandas as pd
from logs import logger
from src.utils import load_config
from typing import Optional
from sklearn.model_selection import train_test_split

# Load configuration
config = load_config()
    
    
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a given path.

    Parameters:
    - data_path (str): The path to the data file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    # Check if the path exists
    if not os.path.exists(data_path):
        logger.error(f"The path '{data_path}' does not exist.")
        raise FileNotFoundError(f"The path '{data_path}' does not exist.")

    # Load the data
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data from '{data_path}': {str(e)}")
        raise Exception(f"Error loading data from '{data_path}': {str(e)}")
    

def save_data(data: pd.DataFrame, data_path: str) -> None:
    """
    Save data to a given path.

    Parameters:
    - data (pd.DataFrame): Data to be saved.
    - data_path (str): The path to save the data.
    """
    try:
        # Save data
        data.to_csv(data_path, index=False)
        logger.info(f"Data saved to '{data_path}'.")
    except Exception as e:
        logger.error(f"Error saving data to '{data_path}': {str(e)}")
        raise Exception(f"Error saving data to '{data_path}': {str(e)}")
    

def sample_data(data: pd.DataFrame, n_samples: int, target: Optional[str] = None) -> pd.DataFrame:
    """
    Sample data.

    Parameters:
    - data (pd.DataFrame): Input data.
    - n_samples (int): Number of samples to generate.
    - target (Optional[str]): Name of the target variable for stratified sampling.

    Returns:
    - pd.DataFrame: Sampled data.
    """
    try:
        # Sample data
        if target:
            sample_data, _ = train_test_split(
                data,
                test_size=(len(data) - n_samples) / len(data),
                stratify=data[target],
                random_state=config["variables"]["random_state"]
            )
        else:
            sample_data = data.sample(n_samples, random_state=config["variables"]["random_state"])

        logger.info(f"Data sampled successfully.")
        return sample_data
    except Exception as e:
        logger.error(f"Error sampling data: {str(e)}")
        raise Exception(f"Error sampling data: {str(e)}")

