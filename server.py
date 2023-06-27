from visualization.ModularVisualization import ModularServer
from visualization.modules.HexGridVisualization import CanvasHexGrid

from agents import Bee, Flower, Hive, Forest
from model import Garden


def Bees_portrayal(agent):
    if agent is None:
        return

    if type(agent) is Bee:
        return {"Shape": "resources/bee_transparent.png", "scale": 1, "Layer": 1, "nectar": agent.nectar, "hive_pos": agent.rel_pos["hive"], "flower_pos": agent.rel_pos["flower"], "best_nectar": agent.best_flower_nectar}

    elif type(agent) is Flower:
        return {"Shape": "resources/flower.png", "scale": 1, "Layer": 1, "nectar": agent.nectar}

    elif type(agent) is Hive:
        return {"Shape": "resources/hive.png", "scale": 1, "Layer": 1, "honey": agent.honey}

    elif type(agent) is Forest:
        return {"Shape": "resources/tree.jpg", "scale": 1, "Layer": 1}

    return {}


canvas_element = CanvasHexGrid(Bees_portrayal, 20, 20, 600, 600)

server = ModularServer(
    Garden, [canvas_element], "Bee Garden", model_params={"N": 10, "width": 20, "height": 20, "num_hives": 1, "num_bouquets": 1, "num_forests": 0, "training": False}
)
# server.launch()
