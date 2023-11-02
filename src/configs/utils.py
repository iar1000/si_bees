import json

def load_config_dict(path: str) -> dict:
    """loads the config json file and returns a dict with the values"""
    with open(path, "r") as f:
        config_dict = json.load(f)
    return config_dict