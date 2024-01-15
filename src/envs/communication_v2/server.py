import os
import shutil
from mesa.visualization.ModularVisualization import ModularServer, TextElement
from mesa.visualization.modules import CanvasGrid, ChartModule
from ray.rllib.algorithms.ppo import PPO
from ray import tune

from envs.communication_v2.agents import Worker, Oracle, Platform
from envs.communication_v2.model import CommunicationV2_model
from envs.communication_v2.environment import CommunicationV2_env

class GamestateTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        out = [
            f"terminated   : {0 == -sum([1 for a in model.schedule.agents if type(a) is Worker and a.output != model.oracle.state])}",
            f"states       : {model.oracle.state} {[a.output for a in model.schedule.agents if type(a) is Worker]}",
        ]
        return "<h3>Status</h3>" + "<br />".join(out)

class GraphElement(TextElement):
    path = "/Users/sega/Code/si_bees/venv/lib/python3.11/site-packages/mesa_viz_tornado/assets"
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    counter = 0
    def render(self, model):
        # delete old images to remove spam
        path_old = os.path.join(self.path, f"graph_{self.counter}.png")
        if os.path.isfile(path_old):
            os.remove(path_old)
        self.counter += 1
        # save new graph image
        path_new = os.path.join(self.path, f"graph_{self.counter}.png")
        model.save_graph(path=path_new)
        return f"<img src='custom/{path_new}'>" 
 
def agent_visualisation(agent):
    colors = ["green", "black", "red", "blue", "orange"]
    if agent is None:
        return
    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.5, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 1}
    if type(agent) is Oracle:
        return {"Shape": "rect", "w": 1, "h": 1, "Color": colors[agent.state % 6], "Filled": "true", "Layer": 0}
    
def create_server(env_config: dict, curr_level: int,
                    model_checkpoint: str):
    tune.register_env("CommunicationV2_env", lambda env_config: CommunicationV2_env(env_config))
    model_params = {}
    model_params["policy_net"] = PPO.from_checkpoint(model_checkpoint) if model_checkpoint else None
    model_params["inference_mode"] = True
    for k, i in env_config["agent_config"].items():
        model_params[k] = i
    for k, i in env_config["model_configs"][str(curr_level)]["model_params"].items():
        model_params[k] = i

    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=model_params["n_tiles_x"], 
        grid_height=model_params["n_tiles_y"], 
        canvas_width=300,
        canvas_height=300)
    
    game_state = GamestateTextElement()
    graph = GraphElement()
    #score = ChartModule([{"label": "score", "Label": "score", "Color": "Black"},],
     #                   data_collector_name='datacollector',
     #                   canvas_height=30, canvas_width=100)
    #points = ChartModule([{"label": "max_obtainable_reward", "Label": "max_obtainable_reward", "Color": "Black", "color": "Black"},
    #                      {"label": "accumulated_obtainable_reward", "Label": "accumulated_obtainable_reward", "Color": "Black", "color": "Black"},],
     #                   data_collector_name='datacollector',
    #                    canvas_height=30, canvas_width=100)

    server = ModularServer(
        CommunicationV2_model, 
        [canvas, game_state, graph], 
        env_config["task_name"], 
        model_params=model_params,
    )

    return server
