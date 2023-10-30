
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.communication_v0.model import CommunicationV0_model

class CommunicationV0_env(MultiAgentEnv):
    """
    base environment to learn communication.
    synchronised actions, all alive agents step simulatiniously
    an oracle outputs information if the agents should step on a particular field. once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    metadata = {
        "render.modes": ["agent_pos"],
    }

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.render_mode = config["render_mode"]
 
        # create underlying mesa model
        self.model = CommunicationV0_model(config)
        self.agents, self.agent_to_id =  self.model.get_possible_agents()
        assert len(self.agents) > 0, "need to have agents in the game"

        # create env state
        self.obss = set()
        self.rewardss = set()
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = self.model.get_obs_space(agent_id=0)
        self.action_space = self.model.get_action_space(agent_id=0)
        print(f"created environment: num_agents={len(self.agents)}, ids:{[self.agent_to_id[a] for a in self.agents]}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # re-create underlying mesa model
        self.model = CommunicationV0_model(self.config)
        self.agents, self.agent_to_id =  self.model.get_possible_agents()
        assert len(self.agents) > 0, "need to have agents in the game"
        
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
        n_next_step, reward = self.model.finish_round()

        # gather new observations
        for agent in self.agents:
            agent_id = self.agent_to_id[agent]
            obs[agent] = self.model.observe_agent(agent_id=agent_id)
            rew[agent] = reward

        # kill simulation after max_round steps
        terminated["__all__"] = False
        truncated["__all__"] = n_next_step >= self.config["max_steps"]

        if self.render_mode == "agent_pos":
            print(_format_move_actions(action_dict=action_dict))
            self.model.print_agent_locations()

        return obs, rew, terminated, truncated, info

def _format_move_actions(action_dict) -> str:
    out = "\t\t"
    for agent, action in action_dict.items():
        out += f"{agent}: ({action[0]}, {action[1]}) "
    return out