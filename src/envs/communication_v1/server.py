import os
import shutil
from mesa.visualization.ModularVisualization import ModularServer, TextElement
from mesa.visualization.modules import CanvasGrid, ChartModule
from ray.rllib.algorithms.ppo import PPO

from envs.communication_v1.agents import Worker, Oracle, Platform
from envs.communication_v1.model import CommunicationV1_model

class GamestateTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        out = [
            f"total_reward      : {model.accumulated_reward}/{model.max_total_reward} (timeout {model.time_to_reward})",
            f"obtainable_reward : {model.accumulated_obtainable_reward}/{model.max_obtainable_reward} (timeout {model.time_to_change})",
            f"score             : {0 if model.max_obtainable_reward <= 0 else model.accumulated_obtainable_reward / model.max_obtainable_reward}",
        ]

        return "<h3>Gamestate</h3>" + "<br />".join(out)

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
        self.counter += 1
        name = f"graph_{self.counter}.png"
        model.save_graph(path=os.path.join(self.path, name))
        return f"<img src='custom/{name}'>" 
 
def agent_visualisation(agent):
    if agent is None:
        return

    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.5, "Color": "black", "Filled": "true", "Layer": 1}
    
    if type(agent) is Platform:
        is_reward, _ = agent.model.compute_reward()

        circ = {"Shape": "circle", "r": 1, "Color": "green", "Filled": "true", "Layer": 0}
        if is_reward == 0:
            circ["Color"] = "orange"
        elif is_reward > 0:
            circ["Color"] = "green"
        else:
            circ["Color"] = "red"
        
        return circ
    
    if type(agent) is Oracle:
        square = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}
        if not agent.is_active():
            square["Color"] = "orange"
        elif agent.get_state() == 1:
            square["Color"] = "green"
        else:
            square["Color"] = "red"
        return square
    
def create_server(env_config: dict, curr_level: int,
                    model_checkpoint: str):
    
    policy_net = None
    if model_checkpoint:
        policy_net = PPO.from_checkpoint(model_checkpoint)

    # merge config dict
    model_params = {}
    model_params["policy_net"] = policy_net
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
    score = ChartModule([{"label": "score", "Label": "score", "Color": "Black"},],
                        data_collector_name='datacollector',
                        canvas_height=30, canvas_width=100)
    points = ChartModule([{"label": "max_obtainable_reward", "Label": "max_obtainable_reward", "Color": "Black", "color": "Black"},
                          {"label": "accumulated_obtainable_reward", "Label": "accumulated_obtainable_reward", "Color": "Black", "color": "Black"},],
                        data_collector_name='datacollector',
                        canvas_height=30, canvas_width=100)

    server = ModularServer(
        CommunicationV1_model, 
        [canvas, game_state, graph, score, points], 
        env_config["task_name"], 
        model_params=model_params,
    )

    return server
