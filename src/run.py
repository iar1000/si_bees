import argparse
import environment as environment
import numpy as np

from experiments import default_config
from envs.communication_v0.server import create_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--task", default="communicationV0", help="name of the task: communicationV0 |")
    parser.add_argument("--visual", default=False, action='store_true', help="starts webserver to run the task with visualization")


    args = parser.parse_args()

    if args.visual:
        server = create_server(default_config)
        server.launch(open_browser=True)

    else:
        env = environment.env(config=default_config, render_mode="minimal")
        env.reset(seed=42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:    
                action = env.action_space(agent).sample(observation["action_mask"])
            
            env.step(action)
