import os
from mesa.visualization.ModularVisualization import ModularServer, TextElement
from mesa.visualization.modules import CanvasGrid
from ray.rllib.algorithms.ppo import PPO
from ray import tune

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
        return {"Shape": "circle", "r": 0.5, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 1}
    if type(agent) is Oracle:
        return {"Shape": "rect", "w": 1, "h": 1, "Color": colors[agent.state % 6], "Filled": "true", "Layer": 0}
    
def create_server(model_checkpoint: str):
    tune.register_env("Simple_env", lambda env_config: Simple_env(env_config))

    config = read_yaml_config(os.path.join(model_checkpoint, "env_config.yaml"))
    
    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=config["model"]["grid_width"], 
        grid_height=config["model"]["grid_height"], 
        canvas_width=300,
        canvas_height=300)
    
    game_state = GamestateTextElement()

    server = ModularServer(
        Simple_model, 
        [canvas, game_state], 
        {}, 
        model_params={
            "config": config,
            "inference_mode": True,
            "policy_net": PPO.from_checkpoint(model_checkpoint) if model_checkpoint else None},
    )

    return server
