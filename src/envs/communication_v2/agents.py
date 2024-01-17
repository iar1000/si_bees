
import mesa
import numpy as np


class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 hidden_vec: np.array,
                 output: int):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"
        self.hidden_vec = hidden_vec
        self.output = output

class Oracle(mesa.Agent):
    """
    oracle that displays which platform to step on
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 state: int):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.state = state

################################################################################################

class Platform(mesa.Agent):
    """
    platform that can be stepped on
    """
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
        self.name = f"platform_{unique_id}"

    def is_occupied(self) -> bool:
        nbs = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=0)
        nbs = [nb for nb in nbs if type(nb) is not Platform]
        return len(nbs) > 0