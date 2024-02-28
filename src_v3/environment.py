from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces.utils import flatdim

from task_models import base_model, lever_pulling_model, moving_history_model_marl, moving_model_marl, mpe_spread_marl_model, transmission_extended_model_marl, transmission_model_marl, transmission_model_rl

RL_ENV = "base_env"
MARL_ENV = "marl_env"

def load_task_model(name: str, env_type) -> base_model:
    """load task model based on task and environment (RL/MARL) type"""
    if name == "lever-pulling":
        if type(env_type) == marl_env or env_type == MARL_ENV:
            return lever_pulling_model
        
    if name == "transmission":
        if type(env_type) == base_env or env_type == RL_ENV:
            return transmission_model_rl
        elif type(env_type) == marl_env or env_type == MARL_ENV:
            return transmission_model_marl
        
    if name == "transmission_extended":
        if type(env_type) == marl_env or env_type == MARL_ENV:
            return transmission_extended_model_marl
    
    if name == "moving":
        if type(env_type) == marl_env or env_type == MARL_ENV:
            return moving_model_marl
    
    if name == "moving_history":
        if type(env_type) == marl_env or env_type == MARL_ENV:
            return moving_history_model_marl
    
    if name == "mpe_spread":
        if type(env_type) == marl_env or env_type == MARL_ENV:
            return mpe_spread_marl_model
            
        
    raise NotImplementedError(f"task {name} is not implemented for environment type {type(env_type)}")

class base_env(TaskSettableEnv):
    """base environment for RL applications"""
    def __init__(self, config: dict,
                 initial_task_level: int = 0):
        super().__init__()

        # load task curriculum
        self.task_name = config["task_model"]
        self.task_configs = [config["task_configs"][task_level] for task_level in config["task_configs"]]
        self.max_task_level = len(self.task_configs) - 1
        
        # set initial task
        self.curr_task_level = initial_task_level
        self.curr_config = self.task_configs[self.curr_task_level]

        # instantiate model
        self.model = load_task_model(self.task_name, self)(config=self.curr_config)
        self.observation_space = self.model.get_obs_space()
        self.action_space = self.model.get_action_space()

        print("\n=== environment ===")
        print(f"env type            = {type(self)}")
        print(f"task model          = {type(self.model)}")
        print(f"size action_space   = {flatdim(self.action_space)}")
        print(f"size obs_space      = {flatdim(self.observation_space)}")
        print(f"tasks levels        = {len(self.task_configs)}")
        print(f"initial task level  = {self.curr_task_level}")
        print()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.model = load_task_model(self.task_name, self)(config=self.curr_config)
        return self.model.get_obs(), {}

    def step(self, actions):
        obs, reward, terminated, truncated = self.model.step(actions=actions)
        return obs, reward, terminated, truncated, {} 

    def get_task(self):
        """get current curriculum task"""
        return self.curr_task_level

    def set_task(self, task: int):
        """set next curriculum task"""
        self.curr_task_level = min(self.max_task_level, task)
        self.curr_config = self.task_configs[self.curr_task_level]

class marl_env(base_env, MultiAgentEnv):
    """base environment for MARL applications"""
    def __init__(self, config: dict, initial_task_level: int = 0):
        super().__init__(config, initial_task_level)