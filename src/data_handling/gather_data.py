import os
import pandas as pd
import numpy as np
from logs import logger
from src.utils import load_config
from typing import Optional, Union
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import train_test_split  

# Instantiate config
config = load_config()

def load_data(data_path: str) -> Union[pd.DataFrame, np.ndarray, csr_matrix]:
    """
    Load data from a given path.

    Parameters:
    - data_path (str): The path to the data file.

    Returns:
    - Union[pd.DataFrame, np.ndarray, csr_matrix]: Loaded data.
    """
    # Check if the path exists
    if not os.path.exists(data_path):
        logger.error(f"The path '{data_path}' does not exist.")
        raise FileNotFoundError(f"The path '{data_path}' does not exist.")

    # Load the data
    try:
        # Check if the file is a CSV
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        # Check if the file is a NumPy file
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        # Check if the file is a CSR matrix (npz)
        elif data_path.endswith('.npz'):
            data = load_npz(data_path)
        else:
            logger.error(f"Unsupported file format for '{data_path}'. "
                         "Supported formats: CSV (.csv), NumPy (.npy), or CSR matrix (.npz)")
            raise ValueError(f"Unsupported file format for '{data_path}'. "
                             "Supported formats: CSV (.csv), NumPy (.npy), or CSR matrix (.npz)")

        return data
    except Exception as e:
        logger.error(f"Error loading data from '{data_path}': {str(e)}")
        raise Exception(f"Error loading data from '{data_path}': {str(e)}")


def save_data(data: Union[pd.DataFrame, pd.Series, np.ndarray, csr_matrix], data_path: str) -> None:
    """
    Save data to a given path.

    Parameters:
    - data (Union[pd.DataFrame,  pd.Series, np.ndarray, csr_matrix]): Data to be saved.
    - data_path (str): The path to save the data.
    """
    try:
        # Check data type and save accordingly
        if isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False)
        elif isinstance(data, np.ndarray):
            np.save(data_path, data)
        else:
            save_npz(data_path, data)

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
