import argparse
import numpy as np

from envs.communication_v0.server import create_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--task", default="communicationV0", help="name of the task: communicationV0 |")

    args = parser.parse_args()

    server = create_server()
    server.launch(open_browser=True)

 
