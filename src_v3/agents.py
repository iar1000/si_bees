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
        
TYPE_MPE_WORKER = 0
TYPE_MPE_LANDMARK = 1

class mpe_worker(mesa.Agent):
    def __init__(self, 
                unique_id: int, 
                model: mesa.Model, 
                n_hidden_states: int,
                size: float = 0.15,
                mass: float = 1,
                max_speed: float = 1):
        super().__init__(unique_id, model)
        self.name = f"agent_{unique_id}"
        self.type = TYPE_MPE_WORKER
        self.hidden_state = np.random.rand(n_hidden_states)
        self.size = size
        self.mass = mass
        self.velocity = np.zeros(2)
        self.max_speed = max_speed

class mpe_landmark(mesa.Agent):
    def __init__(self, 
                unique_id: int, 
                model: mesa.Model, 
                n_hidden_states: int,
                size: float = 0.2,
                mass: float = 1):
        super().__init__(unique_id, model)
        self.name = f"agent_{unique_id}"
        self.type = TYPE_MPE_LANDMARK
        self.hidden_state=np.zeros(n_hidden_states)
        self.size = size
        self.velocity = np.zeros(2)
        self.mass = mass
