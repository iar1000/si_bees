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
    
def create_server(model_checkpoint: str):
    
    policy_net = None
    if model_checkpoint:
        policy_net = PPO.from_checkpoint(model_checkpoint)

    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=11, 
        grid_height=11, 
        canvas_width=500,
        canvas_height=500)

    server = ModularServer(
        CommunicationV0_model, 
        [canvas], 
        "communication v0", 
        model_params=
            {
                "n_agents": 5,
                "agent_placement": "random",
                "plattform_distance": 5,
                "oracle_burn_in": 10,
                "p_oracle_change": 0.05,
                "n_tiles_x": 11,
                "n_tiles_y": 11,
                "max_steps": 100,
                "size_hidden": 8,
                "size_comm": 8,
                "dist_visibility": 2,
                "dist_comm": 2,
                "len_trace": 2,
                "policy_net": policy_net
            }
    )

    return server
