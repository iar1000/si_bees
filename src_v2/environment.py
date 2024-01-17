from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from gymnasium.spaces.utils import flatdim

from model import Simple_model



class Simple_env(TaskSettableEnv):
    
    def __init__(self, config,
                 initial_task_level: int = 0):
        super().__init__()
        
        # handling configs for curriculum
        self.task_level = initial_task_level
        self.curriculum_configs = [config[task] for task in config]
        self.curr_config = self.curriculum_configs[self.task_level]

        self.model = Simple_model(config=self.curr_config)
        self.observation_space = self.model.get_obs_space()
        self.action_space = self.model.get_action_space()
        
        print("\n=== env ===")
        print(f"size action_space   = {flatdim(self.action_space)}")
        print(f"size obs_space      = {flatdim(self.observation_space)}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.model = Simple_model(config=self.curr_config)
        return self.model.get_obs(), {}

    def step(self, actions):
        obs, reward, terminated, truncated = self.model.step(actions=actions)
        return obs, reward, terminated, truncated, {} 

    def get_task(self):
        """get current curriculum task"""
        return self.task_level

    def set_task(self, task: int):
        """set next curriculum task"""
        self.task_level = min(len(self.configs) - 1, task)