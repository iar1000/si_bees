from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization import ChartModule

from envs.communication_v0.agents import Worker, Oracle, Plattform
from envs.communication_v0.model import CommunicationV0_model


def agent_visualisation(agent):
    if agent is None:
        return

    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.5, "Color": "black", "Filled": "true", "Layer": 1}
    
    if type(agent) is Plattform:
        oracle, _ = agent.model.get_oracle_and_plattform()
        time_to_reward = agent.model.time_to_reward

        circ = {"Shape": "circle", "r": 1, "Color": "green", "Filled": "true", "Layer": 0}
        if time_to_reward > 0:
            circ["Color"] = "orange"
        elif oracle.get_state() == 1 and len(agent.get_occupants()) > 0 or \
            oracle.get_state() == 0 and len(agent.get_occupants()) <= 0:
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
    
def create_server(config):
    chart = ChartModule(
        [{"Label": "Gini", "Color": "Black"}], data_collector_name="datacollector"
    )

    canvas = CanvasGrid(
        agent_visualisation, 
        config["mesa_grid_width"], config["mesa_grid_height"], 
        config["mesa_grid_width"] * config["mesa_tile_size"], config["mesa_grid_height"] * config["mesa_tile_size"])

    server = ModularServer(
        CommunicationV0_model, 
        [canvas], 
        "communication v0", 
        model_params=
            {"config": config, 
            }
    )

    return server
