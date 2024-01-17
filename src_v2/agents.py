
import mesa
import numpy as np


class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 output: int):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"
        self.output = output

class Oracle(mesa.Agent):
    """
    oracle that displays which platform to step on
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.state = state