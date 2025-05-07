"""
Configuration loader for the CTC Speech Refinement project.

This module provides functions for loading and managing configuration settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config_from_module(module_path: str) -> Dict[str, Any]:
    """
    Load configuration from a Python module.
    
    Args:
        module_path: Path to the Python module.
        
    Returns:
        Dictionary containing configuration settings.
    """
    logger.info(f"Loading configuration from module: {module_path}")
    
    # Load module
    spec = importlib.util.spec_from_file_location("config_module", module_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract configuration
    config = {}
    for key in dir(config_module):
        if not key.startswith("__") and not key.startswith("_"):
            config[key] = getattr(config_module, key)
    
    return config

def load_config_from_json(json_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        Dictionary containing configuration settings.
    """
    logger.info(f"Loading configuration from JSON: {json_path}")
    
    with open(json_path, "r") as f:
        config = json.load(f)
    
    return config

def save_config_to_json(config: Dict[str, Any], json_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Dictionary containing configuration settings.
        json_path: Path to save the JSON file.
    """
    logger.info(f"Saving configuration to JSON: {json_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save configuration
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Dictionary containing default configuration settings.
    """
    logger.info("Loading default configuration")
    
    # Get path to default config module
    default_config_path = os.path.join(os.path.dirname(__file__), "default_config.py")
    
    # Load configuration
    return load_config_from_module(default_config_path)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration dictionary to override base.
        
    Returns:
        Merged configuration dictionary.
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override or add value
            merged_config[key] = value
    
    return merged_config

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use default.
    
    Args:
        config_path: Path to configuration file. If None, use default.
        
    Returns:
        Dictionary containing configuration settings.
    """
    # Get default configuration
    config = get_default_config()
    
    # Override with user configuration if provided
    if config_path:
        if config_path.endswith(".py"):
            user_config = load_config_from_module(config_path)
        elif config_path.endswith(".json"):
            user_config = load_config_from_json(config_path)
        else:
            logger.warning(f"Unsupported configuration file format: {config_path}")
            return config
        
        # Merge configurations
        config = merge_configs(config, user_config)
    
    return config
