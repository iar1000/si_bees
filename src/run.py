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

    parser.add_argument("--task", default="communicationV0", help="name of the task: communicationV0 |")
    parser.add_argument("--run_name", default=None, help="folder name with all checkpoints stored")
    parser.add_argument("--checkpoint", default=None, help="number of checkpoint")
    parser.add_argument("--curr_level", default=0, help="curriculum level of the checkpoint")
    parser.add_argument('--env_config', default="env_comv0.json", help="name of config, should be placed in checkpoints folder")

    args = parser.parse_args()

    checkpoints_dir = "checkpoints"
    cp = None
    env_config = None
    if args.checkpoint:
        cp = os.path.join(checkpoints_dir, args.run_name, zero_pad_to_six(args.checkpoint))
        env_config = load_config_dict(os.path.join(checkpoints_dir, args.run_name, args.env_config))
        
    server = create_server(env_config=env_config, model_checkpoint=cp, curr_level=args.curr_level)
    server.launch(open_browser=True)

 