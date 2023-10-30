
import functools
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.train import report

from envs.communication_v0.model import CommunicationV0_model
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import (
    ActionType,
    AECEnv,
    AECIterable,
    AECIterator,
    AgentID,
    ObsType,
)

def env(render_mode=None, task="communication_v0", config={}):
    """
    returns functional aec environment, wrapped in helpful wrappers
    """    
    # select desired environment
    if task == "communication_v0":
        env = CommunicationV0_env(config, render_mode=render_mode)
    else:
        print("unkown task, could not find environment")
        quit()
    
    # add standart wrappers
    env = wrappers.OrderEnforcingWrapper(env)

    return env

class CommunicationV0_env(AECEnv):
    """
    base environment to learn communication.
    an oracle outputs information if the agents should step on a particular field. once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """
    metadata = {"render_modes": ["minimal"], "name": "communication_v0"}

    def __init__(self, config, render_mode=None):
        """
        from AECEnv, init() has to initialize:
        - possible_agents
        - render_mode (from gym environment)
        """
        # config
        self.config = config

        # setup mesa model
        self.model = CommunicationV0_model(config)
        self.possible_agents, self.agent_to_id =  self.model.get_possible_agents()

        # setup synchrounous run 
        self.buffer_actions = self.config["apply_actions_synchronously"]
        self.action_buffer = dict() # {agent_id: action}

        self.render_mode = render_mode

        self.n_episode = 0


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gymnasium.spaces.Space:
        """
        return action space for the given agent
        the obsercation space has the form of a dict{observation, action_mask}
        """ 
        agent_id = self.agent_to_id[agent]
        return self.model.get_obs_space(agent_id=agent_id)
        
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gymnasium.spaces.Space:
        """
        return action space for the given agent
        the size of the action space can be different than the one in observation[action_mask], 
            but it needs to be clear what is being masked in the custom model
        """
        agent_id = self.agent_to_id[agent]
        return self.model.get_action_space(agent_id=agent_id)


    def observe(self, agent) -> dict:
        """
        return observation of the given agent (can be outdated)
        an observation has the form of a dict{observation, action_mask}
        """
        if not self.observations[agent]:
            self.observations[agent] = self.model.observe_agent(self.agent_to_id[agent])
        return self.observations[agent]


    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        elif self.render_mode == "minimal":
            self.model.print_status()


    def reset(self, seed=None, options=None):
        """
        from AECEnv: reset has to set:
        - possible_agents
        - render_mode (from gym environment)        
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        and must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.n_episode = self.n_episode + 1
        self.model = CommunicationV0_model(config=self.config)
        self.agents = self.possible_agents[:]
        self.observations = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # allows easy cyclic stepping through the agents list
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action) -> None:
        """
        from AECEnv: apply action for the current agent (specified by agent_selection), update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        # remove dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos and skip to the next dead agent or, if none, to next alive one
        if (self.terminations[self.agent_selection] or 
            self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        # get agent
        agent = self.agent_selection
        is_last = self._agent_selector.is_last()
        # self._cumulative_rewards[self.agent_selection] = 0 @todo: when to do this?

        # add action to buffer and progress the simulation if necessary
        self.action_buffer[agent] = action
        if not self.buffer_actions or is_last:
            self.progress_simulation()

        if is_last:
            next_round, reward = self.model.finish_round()
            self.rewards = {agent: reward for agent in self.agents}
            self._accumulate_rewards()
            
            # kill the game after max_rounds
            if next_round >= self.config["mesa_max_rounds"]:
                for a in self.agents:
                    self.truncations[a] = True
            # render
            if self.render_mode:
                self.render()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self._deads_step_first()


    def progress_simulation(self) -> None:
        """
        execute all actions in the action buffer and update the observations of the agents accordingly
        finally clear the action_buffer
        """
        # step
        for agent in self.action_buffer.keys():
            agent_id = self.agent_to_id[agent]
            self.model.step_agent(agent_id=agent_id, action=self.action_buffer[agent])
        
        # update observations
        for agent in self.action_buffer.keys():
            agent_id = self.agent_to_id[agent]
            self.observations[agent] = self.model.observe_agent(agent_id=agent_id)

        self.action_buffer.clear()
