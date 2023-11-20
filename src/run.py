import argparse
import os
import numpy as np

from envs.communication_v0.server import create_server
from configs.utils import load_config_dict


def zero_pad_to_six(number):
    base = "checkpoint_"
    padded_string = str(number).zfill(6)
    return base + padded_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--name", default=None, help="folder name with all checkpoints stored")
    parser.add_argument("--task", default=0, help="curriculum level of the environment")
    parser.add_argument('--env_config', default="env_comv0.json", help="name of config, should be placed in checkpoints folder")

    args = parser.parse_args()

    checkpoints_dir = "checkpoints"
    if args.name:
        cp = os.path.join(checkpoints_dir, args.name)
        env_config = load_config_dict(os.path.join(checkpoints_dir, args.name, args.env_config))
        
    server = create_server(env_config=env_config, model_checkpoint=cp, curr_level=args.task)
    server.launch(open_browser=True)

 