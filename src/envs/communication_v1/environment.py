from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from gymnasium.spaces.utils import flatdim

from envs.communication_v1.model import CommunicationV1_model


class CommunicationV1_env(TaskSettableEnv):
    """
    base environment to learn communication.
    synchronised actions, all alive agents step simulatiniously. 
    collect info as a graph, that can be used for pytorch_geometric library.
    """

    metadata = {
        "render.modes": ["silent", "agent_pos", "status"],
    }

    def __init__(self, config):
        super().__init__()

        # configs
        self.agent_config = config["agent_config"]
        self.model_configs = config["model_configs"]
        self.render_mode = config["render_mode"]

        # curriculum learning
        self.max_steps = config["max_steps"]
        self.curriculum_learning = config["curriculum_learning"]
        self.curr_task = 0
        self.max_task = len(self.model_configs.items()) - 1

        # model
        self.model = self._create_model()
        self.observation_space = self.model.get_obs_space()
        self.action_space = self.model.get_action_space()
        
        print("\n=== env ===")
        print(f"size action_space={flatdim(self.action_space)}")
        print(f"size obs_space={flatdim(self.observation_space)}")
        print()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.model = self._create_model()            
        return self.model.get_obs(), {}

    def step(self, actions):
        self.model.apply_actions(actions)
        reward, is_terminated = self.model.finish_round()
        obs = self.model.get_obs()

        if self.render_mode == "agent_pos":
            self.model.print_agent_locations()
        elif self.render_mode == "status":
            self.model.print_status()

        return obs, reward, is_terminated, is_terminated, {} 

    def get_task(self):
        """get current curriculum task"""
        return self.curr_task

    def set_task(self, task: int):
        """set next curriculum task"""
        self.curr_task = min(task, self.max_task)

    def _curr_model_config(self) -> [str, dict]:
        """fetches and returns the name and config dict for the mesa model of the current task"""
        task = self.model_configs[str(self.curr_task)]
        return task["description"], task["model_params"] 

    def _create_model(self):
        """creates a mesa model based on the curriculum level and agent configs"""
        # merge the fixed agent config with the task dependend model config
        # add the max_steps, as it is a parameter of the model
        parameter_dict = {}
        _, model_config = self._curr_model_config()
        for d in [self.agent_config, model_config, {"max_steps":self.max_steps}]:
            for k, v in d.items():
                parameter_dict[k] = v

        return CommunicationV1_model(**parameter_dict)