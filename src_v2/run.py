import argparse
import os

from server import create_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--checkpoint", default=None, help="folder name where checkpoints is stored")
    args = parser.parse_args()

    checkpoint = os.path.join("checkpoints", args.checkpoint) if args.checkpoint else None
    server = create_server(model_checkpoint=checkpoint)
    
    server.launch(open_browser=True)

 