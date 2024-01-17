from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from gymnasium.spaces.utils import flatdim

from model import Simple_model



class Simple_env(TaskSettableEnv):
    
    def __init__(self, config):
        super().__init__()

        self.model = Simple_model(config=config)
        self.observation_space = self.model.get_obs_space()
        self.action_space = self.model.get_action_space()
        
        print("\n=== env ===")
        print(f"size action_space   = {flatdim(self.action_space)}")
        print(f"size obs_space      = {flatdim(self.observation_space)}")
        print()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.model = Simple_model(**self.model_params)
        return self.model.get_obs(), {}

    def step(self, actions):
        obs, reward, terminated, truncated = self.model.step(actions=actions)
        return obs, reward, terminated, truncated, {} 

    def get_task(self):
        """get current curriculum task"""
        return self.curr_task

    def set_task(self, task: int):
        """set next curriculum task"""
        self.curr_task = min(task, 0)