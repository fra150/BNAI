# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "src/utils/config.yaml") -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing the configuration parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a valid YAML dictionary")
            
        # Validate required configuration sections
        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Missing required configuration section: {section}")
                
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model-specific configuration."""
    return config.get("model", {})

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training-specific configuration."""
    return config.get("training", {})

def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data-specific configuration."""
    return config.get("data", {})

if __name__ == "__main__":
    try:
        cfg = load_config()
        print("Model config:", get_model_config(cfg))
        print("Training config:", get_training_config(cfg))
        print("Data config:", get_data_config(cfg))
    except Exception as e:
        print(f"Error: {str(e)}")