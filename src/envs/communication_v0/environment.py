
from typing import List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType

from envs.communication_v0.model import CommunicationV0_model

class CommunicationV0_env(MultiAgentEnv, TaskSettableEnv):
    """
    base environment to learn communication.
    synchronised actions, all alive agents step simulatiniously
    an oracle outputs information if the agents should step on a particular field. once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    metadata = {
        "render.modes": ["silent", "agent_pos", "status"],
    }

    def __init__(self, config):
        super().__init__()

        self.agent_config = config["agent_config"]
        self.model_configs = config["model_configs"]
        self.render_mode = config["render_mode"]

        # curriculum learning
        self.max_steps = config["max_steps"]
        self.curriculum_learning = config["curriculum_learning"]
        self.curr_task = 0
        self.max_task = len(self.model_configs.items()) - 1

        self.model = self._create_model()
        self.agents, self.agent_to_id =  self.model.get_possible_agents()
        self.observation_space = self.model.get_obs_space(agent_id=0)
        self.action_space = self.model.get_action_space(agent_id=0)

        # create env state
        self.obss = set()
        self.rewardss = set()
        self.terminateds = set()
        self.truncateds = set()
        #self.observation_space = self.model.get_obs_space(agent_id=0)
        #self.action_space = self.model.get_action_space(agent_id=0)
        #print(f"created environment: num_agents={len(self.agents)}, ids:{[self.agent_to_id[a] for a in self.agents]}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # re-create underlying mesa model, assumes number of agents stay the same
        self.model = self._create_model()
        self.terminateds = set()
        self.truncateds = set()

        obs = {}
        for a in self.agents:
            obs[a] = self.model.observe_agent(self.agent_to_id[a])

        return obs, {}

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

        # step all agents
        for agent, action in action_dict.items():
            agent_id = self.agent_to_id[agent]
            self.model.step_agent(agent_id=agent_id, action=action)
        reward, is_terminated = self.model.finish_round()

        # gather new observations
        for agent in self.agents:
            agent_id = self.agent_to_id[agent]
            obs[agent] = self.model.observe_agent(agent_id=agent_id)
            rew[agent] = reward

        # kill simulation after max_round steps
        terminated["__all__"] = False
        truncated["__all__"] = is_terminated

        if self.render_mode == "agent_pos":
            self.model.print_agent_locations()
        if self.render_mode == "status":
            self.model.print_status()

        return obs, rew, terminated, truncated, info
    

    def get_task(self):
        """get current curriculum task"""
        return self.curr_task

    def set_task(self, task: int):
        """set next curriculum task"""
        self.curr_task = min(task, self.max_task)


    def _curr_model_config(self) -> [str, dict]:
        """fetches and returns the name and config dict for the mesa model of the current task"""
        task = self.model_configs[str(self.curr_task)]
        return task["task_name"], task["model_params"] 

    def _create_model(self):
        """creates a mesa model based on the curriculum level and agent configs"""
        # merge the fixed agent config with the task dependend model config
        # add the max_steps, as it is a parameter of the model
        parameter_dict = {}
        _, model_config = self._curr_model_config()
        for d in [self.agent_config, model_config, {"max_steps":self.max_steps}]:
            for k, v in d.items():
                parameter_dict[k] = v

        return CommunicationV0_model(**parameter_dict)