import argparse
import os

from envs.communication_v2.server import create_server
from configs.utils import load_config_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--name", default=None, help="folder name where checkpoints is stored")
    parser.add_argument("--task", default=0, help="curriculum level of the environment")
    parser.add_argument('--env_config', default="env_comv1_1.json", help="config file for the environment")

    args = parser.parse_args()

    checkpoints_dir = "checkpoints"
    cp = None
    if args.name:
        cp = os.path.join(checkpoints_dir, args.name)
    
    configs_dir = os.path.join("src", "configs")
    env_config = load_config_dict(os.path.join(configs_dir, args.env_config))

    server = create_server(env_config=env_config, model_checkpoint=cp, curr_level=args.task)
    server.launch(open_browser=True)

 