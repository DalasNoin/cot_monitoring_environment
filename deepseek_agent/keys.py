import yaml
import os
from typing import Tuple

# Define the path to the keys file relative to this script (keys.py)
KEYS_FILE_PATH = os.path.join(os.path.dirname(__file__), "keys.yaml")


def _load_keys_from_yaml():
    """Loads keys from the keys.yaml file located next to this script."""
    if not os.path.exists(KEYS_FILE_PATH):
        # Provide a clear error message if the file isn't found where expected
        raise FileNotFoundError(
            f"keys.yaml not found at the expected location: {KEYS_FILE_PATH}. "
            f"Please ensure it is in the same directory as keys.py."
        )

    try:
        with open(KEYS_FILE_PATH, "r") as f:
            keys = yaml.safe_load(f)
            if keys is None:
                print(f"Warning: Keys file at {KEYS_FILE_PATH} is empty.")
                return {}
            return keys
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {KEYS_FILE_PATH}: {e}")
        # Decide how to handle parsing errors, e.g., return empty dict or raise
        raise ValueError(f"Error parsing YAML file {KEYS_FILE_PATH}: {e}") from e


_config = _load_keys_from_yaml()


def load_google_cse_keys() -> Tuple[str, str]:
    """Loads Google API Key and CSE ID from the loaded YAML config."""
    # Use .get() with default None for safer access
    api_key = _config.get("GOOGLE_API_KEY")  # Match the exact key name in your YAML
    cse_id = _config.get("GOOGLE_CSE_ID")  # Match the exact key name in your YAML

    # Check if keys were found and are not placeholders
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError(f"GOOGLE_API_KEY not found or not set in {KEYS_FILE_PATH}")
    if not cse_id or cse_id == "YOUR_GOOGLE_CSE_ID_HERE":
        raise ValueError(f"GOOGLE_CSE_ID not found or not set in {KEYS_FILE_PATH}")

    return api_key, cse_id


def load_deepseek_api_key() -> str:
    """Loads DeepSeek API Key from the loaded YAML config."""
    # Use .get() with default None for safer access
    api_key = _config.get("DEEPSEEK_API_KEY")  # Match the exact key name in your YAML

    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
        raise ValueError(f"DEEPSEEK_API_KEY not found or not set in {KEYS_FILE_PATH}")
    return api_key


# Add other key-loading functions as needed, following the pattern above
