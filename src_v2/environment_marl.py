from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces.utils import flatdim

from model_marl import RELATIVE_STATE_MODELS, SIMPLE_MODELS, Marl_model, Relstate_Model



class Marl_env(TaskSettableEnv, MultiAgentEnv):
    
    def __init__(self, config,
                 env_config_file: str,
                 initial_task_level: int = 0):
        super().__init__()
        
        # handling configs for curriculum
        self.env_config_file = env_config_file
        self.curriculum_configs = [config[task] for task in config]
        self.max_task_level = len(self.curriculum_configs) - 1
        self.task_level = initial_task_level
        self.curr_config = self.curriculum_configs[self.task_level]

        self.model = Marl_model(config=self.curr_config) if self.env_config_file in SIMPLE_MODELS \
            else Relstate_Model(config=self.curr_config) if self.env_config_file in RELATIVE_STATE_MODELS else None
        self.observation_space = self.model.get_obs_space()
        self.action_space = self.model.get_action_space()
        
        print("\n=== env ===")
        print(f"size action_space   = {flatdim(self.action_space)}")
        print(f"size obs_space      = {flatdim(self.observation_space)}")
        print(f"num_tasks           = {len(self.curriculum_configs)}")
        print(f"initial_task        = {self.task_level}")
        print()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.model = Marl_model(config=self.curr_config) if self.env_config_file in SIMPLE_MODELS \
            else Relstate_Model(config=self.curr_config) if self.env_config_file in RELATIVE_STATE_MODELS else None
        return self.model.get_obss(), {}

    def step(self, actions):
        obs, reward, terminated, truncated = self.model.step(actions=actions)
        return obs, reward, terminated, truncated, {} 

    def get_task(self):
        """get current curriculum task"""
        return self.task_level

    def set_task(self, task: int):
        """set next curriculum task"""
        self.task_level = min(self.max_task_level, task)
        self.curr_config = self.curriculum_configs[self.task_level]
