
from ray import tune
import yaml

def create_tunable_config(config: dict):
    """read in config dictionary and convert min-max ranges of keys to tunable parameters"""
    tunable_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            if isinstance(v["min"], int) and isinstance(v["max"], int):
                tunable_config[k] = tune.choice(list(range(v["min"], v["max"] + 1)))
            else:
                tunable_config[k] = tune.uniform(v["min"], v["max"])       
        elif isinstance(v, list):
            tunable_config[k] = tune.choice(v)
        else:
            tunable_config[k] = v
    return tunable_config

def read_yaml_config(path: str):
    """read in and convert yaml file to dict"""
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File not found: {path}")