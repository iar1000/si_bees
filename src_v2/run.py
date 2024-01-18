import argparse
import os
import os
from ray import tune
from mesa.visualization.ModularVisualization import ModularServer, TextElement
from mesa.visualization.modules import CanvasGrid
from ray.rllib.algorithms.ppo import PPO

from model import Simple_model
from agents import Oracle, Worker
from environment import Simple_env
from utils import read_yaml_config

class GamestateTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        out = [
            f"terminated   : {0 == -sum([1 for a in model.schedule.agents if type(a) is Worker and a.output != model.oracle.state])}",
            f"states       : {model.oracle.state} {[a.output for a in model.schedule.agents if type(a) is Worker]}",
        ]
        return "<h3>Status</h3>" + "<br />".join(out)
 
def agent_visualisation(agent):
    colors = ["green", "black", "red", "blue", "orange", "yellow"]
    if agent is None:
        return
    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.9, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 1}
    if type(agent) is Oracle:
        return {"Shape": "rect", "w": 1, "h": 1, "Color": colors[agent.state % 6], "Filled": "true", "Layer": 0}
    
def create_server(model_checkpoint: str, env_config: str, task_level: int):

    config = read_yaml_config(env_config)
    curriculum_configs = [config[task] for task in config]
    task_config = curriculum_configs[task_level]
    tune.register_env("Simple_env", lambda _: Simple_env(config, initial_task_level=task_level))

    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=task_config["model"]["grid_width"], 
        grid_height=task_config["model"]["grid_height"], 
        canvas_width=300,
        canvas_height=300)
    
    game_state = GamestateTextElement()

    server = ModularServer(
        Simple_model, 
        [canvas, game_state], 
        "swarm intelligence with graph networks", 
        model_params={
            "config": task_config,
            "inference_mode": True,
            "policy_net": PPO.from_checkpoint(model_checkpoint) if model_checkpoint else None},
    )

    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--checkpoint",         default=None, help="folder name where checkpoints is stored")
    parser.add_argument('--env_config',         default=None, help="path to env config")
    parser.add_argument('--task_level',         default=0, help="task level of curriculum")
    args = parser.parse_args()

    checkpoint = os.path.join("checkpoints", args.checkpoint) if args.checkpoint else None
    env_config = os.path.join("src_v2", "configs", args.env_config)
    server = create_server(model_checkpoint=checkpoint, env_config=env_config, task_level=args.task_level)
    
    server.launch(open_browser=True)

 