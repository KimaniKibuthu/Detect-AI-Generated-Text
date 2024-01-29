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
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from '{config_path}': {str(e)}")
        raise Exception(f"Error loading configuration from '{config_path}': {str(e)}")

def save_config(config: dict, config_path: str = "configs/config.yaml") -> None:
    """
    Save configuration to a YAML file.

    Parameters:
    - config (dict): Configuration to be saved.
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - None
    """
    try:
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        logger.info(f"Configuration saved to '{config_path}'.")
    except Exception as e:
        logger.error(f"Error saving configuration to '{config_path}': {str(e)}")
        raise Exception(f"Error saving configuration to '{config_path}': {str(e)}")

