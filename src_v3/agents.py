import mesa
import numpy as np

TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

class BaseAgent(mesa.Agent):
    def __init__(self, 
                 unique_id: int, 
                 name_prefix: str, 
                 type: int,
                 model: mesa.Model, 
                 output: int,
                 hidden_state: np.array):
        super().__init__(unique_id, model)
        self.name = f"{name_prefix}_{unique_id}"
        self.type = type
        self.output = output
        self.hidden_state = hidden_state

class Worker(BaseAgent):
    def __init__(self, 
                 unique_id: int, 
                 model: mesa.Model, 
                 output: int,
                 n_hidden_states: int):
        super().__init__(
            unique_id=unique_id, 
            name_prefix="worker",
            type=TYPE_WORKER,
            model=model, 
            output=output, 
            hidden_state=np.random.rand(n_hidden_states))

class Oracle(BaseAgent):
    def __init__(self, 
                 unique_id: int, 
                 model: mesa.Model, 
                 output: int,
                 n_hidden_states: int):
        super().__init__(
            unique_id=unique_id, 
            name_prefix="oracle",
            type=TYPE_ORACLE,
            model=model, 
            output=output, 
            hidden_state=np.zeros(n_hidden_states))