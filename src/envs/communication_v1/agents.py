
import mesa
import numpy as np

    
class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model, comm_vec: np.array):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"
        self.comm_vec = comm_vec
    
    def set_comm_vec(self, comm_vec):
        self.comm_vec = comm_vec

    def get_comm_vec(self):
        return self.comm_vec

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


class Oracle(mesa.Agent):
    """
    oracle that displays which platform to step on
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 is_active: bool = False, 
                 state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.active = is_active
        self.state = state

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def is_active(self):
        return self.active
    
    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state