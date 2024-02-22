import random
import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete, flatten
from ray.rllib.algorithms import Algorithm
import mesa
import numpy as np
import torch
from ray.rllib.policy.sample_batch import SampleBatch

GRAPH_HASH = 1000000

class Marl_Lever_Pulling():

    def __init__(self, config: dict,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        self.n_agents = 5
        self.agent_id_max = 500
        self.n_levers = 5

        self.reward_total = 0
        self.reward_lower_bound = 0
        self.reward_upper_bound = 0
        self.n_state_switches = 0

        self.grid = mesa.space.MultiGrid(1, 1, False)

        # inference mode
        self.policy_net = policy_net
        self.inference_mode = inference_mode
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(self.n_levers),                             # output
        ])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        graph_state = Box(0, int(GRAPH_HASH), shape=(1,), dtype=np.float32) # current step hash
        agent_states = Tuple([Tuple([
            Discrete(2),                     # active flag
            Discrete(self.agent_id_max)      # agent id
        ]) for _ in range(self.n_agents)])
        edge_states = Tuple([Tuple([
            Discrete(2),                            # exists flag
            Box(0, 1, shape=(2,), dtype=np.float32), # relative position to the given node
        ]) for _ in range(self.n_agents * self.n_agents)])
        
        return Tuple([graph_state, agent_states, edge_states])
    
    def get_obss(self):
        step_hash = np.array([random.randint(0, GRAPH_HASH)])
        agent_ids = [random.randint(0, self.agent_id_max - 1) for _ in range(self.n_agents)]
        agent_states = [tuple([0, agent_ids[i]]) for i in range(self.n_agents)]
        edge_states = [tuple([1, np.array([0,0])]) for _ in range(self.n_agents ** 2)]
        
        obss = dict()
        for i in range(self.n_agents):
            curr_agent_state = agent_states.copy()
            curr_agent_state[i] = tuple([1, agent_ids[i]])
            obss[i] = tuple([step_hash, tuple(curr_agent_state), tuple(edge_states)])
        return obss
    
    def step(self, actions=None):

        # determine actions for inference mode
        if self.inference_mode:
            actions = dict()
            obss = self.get_obss()
            if self.policy_net:
                for i in range(self.n_agents):
                    actions[i], _, _ = self.policy_net.compute_single_action(obss[i], state=np.array([i, self.n_agents]))
            else:
                for i in range(self.n_agents):
                    actions[i] = self.get_action_space().sample()

        levers = actions.values()
        r = float(len(set(levers))) / self.n_levers
        self.reward_total = r
        self.reward_upper_bound = 1

        if self.inference_mode:
            print("pulled levers: ")
            print(levers)
            print("reward: ", r)
        return self.get_obss(), {a: r for a in range(self.n_agents)}, {"__all__": True}, {"__all__": False}