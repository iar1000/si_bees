import argparse
import numpy as np

from envs.communication_v0.server import create_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--task", default="communicationV0", help="name of the task: communicationV0 |")
    parser.add_argument("--checkpoint", default=None, help="path to model checkpoint")

    args = parser.parse_args()

    server = create_server(model_checkpoint=args.checkpoint)
    server.launch(open_browser=True)

 
