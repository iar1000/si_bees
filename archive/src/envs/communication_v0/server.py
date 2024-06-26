from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization import ChartModule

from ray.rllib.algorithms.ppo import PPO

from envs.communication_v0.agents import Worker, Oracle, Plattform
from envs.communication_v0.model import CommunicationV0_model


def agent_visualisation(agent):
    if agent is None:
        return

    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.5, "Color": "black", "Filled": "true", "Layer": 1}
    
    if type(agent) is Plattform:
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

# @todo: refactor to make it compatible with multiple envs
def create_server(curr_level: int, env_config: dict, model_checkpoint: str):
    
    policy_net = None
    if model_checkpoint:
        policy_net = PPO.from_checkpoint(model_checkpoint)

    # merge config dict
    model_params = {}
    model_params["max_steps"] = env_config["env_config"]["max_steps"]
    model_params["policy_net"] = policy_net
    for k, i in env_config["env_config"]["agent_config"].items():
        model_params[k] = i
    for k, i in env_config["env_config"]["model_configs"][str(curr_level)]["model_params"].items():
        model_params[k] = i

    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=model_params["n_tiles_x"], 
        grid_height=model_params["n_tiles_y"], 
        canvas_width=500,
        canvas_height=500)

    server = ModularServer(
        CommunicationV0_model, 
        [canvas], 
        env_config["task_name"], 
        model_params=model_params
    )

    return server
