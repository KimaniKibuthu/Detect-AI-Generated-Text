"""
Contains the utility functions for the project
"""
import yaml
from logs import logger

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Loaded configuration.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from '{config_path}': {str(e)}")
        raise Exception(f"Error loading configuration from '{config_path}': {str(e)}")

