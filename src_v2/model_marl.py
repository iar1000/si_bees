from math import floor
import mesa
import random
import numpy as np
import networkx as nx
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete
import torch
from agents import Oracle, Worker

from utils import compute_agent_placement, get_relative_pos

MAX_DISTANCE = 100
GRAPH_HASH = 1000000

TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

MODEL_TYPE_SIMPLE = 0
MODEL_TYPE_MOVING = 1
SIMPLE_MODELS = ["env_config_0.yaml",
                    "env_config_1.yaml",
                    "env_config_2.yaml",
                    "env_config_3.yaml",
                    "env_config_4.yaml",
                    "env_config_5.yaml",
                    "env_config_6.yaml"]
MOVING_MODELS = ["env_config_7.yaml",
                 "env_config_8.yaml"]

class Marl_model(mesa.Model):
    """
    the oracle outputs a number which the agents need to copy
    the visibility range is limited, so the agents need to communicate the oracle output
    reward: how many agents have the correct output
    """

    def __init__(self, config: dict,
                 model_type: int = MODEL_TYPE_SIMPLE,
                 use_cuda: bool = False,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__()
        
        self.model_type = model_type
        # workers
        self.n_workers = config["model"]["n_workers"]
        self.n_hidden_states = config["model"]["n_hidden_state"]
        self.communication_range = config["model"]["communication_range"]
        self.worker_placement = config["model"]["worker_placement"]
        self.worker_init = config["model"]["worker_init"]
        self.reward_calculation = config["model"]["reward_calculation"]
        self.n_agents = self.n_workers + 1
        # oracle
        self.n_oracle_states = config["model"]["n_oracle_states"]
        self.p_oracle_change = config["model"]["p_oracle_change"]
        # grid
        self.grid_size = config["model"]["grid_size"]
        self.grid_middle = floor(self.grid_size / 2)
        # misc
        self.episode_length = config["model"]["episode_length"]
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # tracking
        self.ts_episode = 0
        self.ts_curr_state = 0
        self.state_switch_pause = floor(self.grid_middle / self.communication_range) + 1
        self.n_state_switches = 1
        self.reward_total = 0
        self.reward_lower_bound = 0
        self.reward_upper_bound = 0
        
        # mesa setup
        self.grid = mesa.space.MultiGrid(self.grid_size, self.grid_size, False)
        self.schedule_all = mesa.time.BaseScheduler(self)
        self.schedule_workers = mesa.time.BaseScheduler(self)
        self.current_id = 0

        # initialisation outputs of agents
        oracle_state = random.randint(0, self.n_oracle_states-1)
        if self.worker_init == "uniform":
            r = random.randint(0, self.n_oracle_states-1)
            while r == oracle_state:
                r = random.randint(0, self.n_oracle_states-1)
            worker_output = [r for _ in range(self.n_workers)]
        else:
            worker_output = [random.randint(0, self.n_oracle_states-1) for _ in range(self.n_workers)]

        # place agents
        self.oracle = Oracle(self._next_id(), self, state=oracle_state)
        oracle_pos = (self.grid_middle, self.grid_middle)
        self.grid.place_agent(agent=self.oracle, pos=oracle_pos)
        self.schedule_all.add(self.oracle)
        agent_positions = compute_agent_placement(self.n_workers, self.communication_range, 
                                                  self.grid_size, self.grid_size, 
                                                  oracle_pos, self.worker_placement)
        for i, curr_pos in enumerate(agent_positions):
            worker = Worker(self._next_id(), self, output=worker_output[i], n_hidden_states=self.n_hidden_states)
            self.grid.place_agent(agent=worker, pos=curr_pos)
            self.schedule_all.add(worker)
            self.schedule_workers.add(worker)

        # inference mode
        self.inference_mode = inference_mode
        self.policy_net = policy_net

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def _compute_reward(self) -> [int, int, int, int]:
        assert self.reward_calculation in {"individual", "per-agent"}

        # compute reward
        n_wrongs = sum([1 for a in self.schedule_workers.agents if a.output != self.oracle.state])
        if self.reward_calculation == "individual":
            rewards = [-n_wrongs for _ in self.schedule_workers]
        elif self.reward_calculation == "per-agent":
            rewards = [1 if a.output == self.oracle.state else -1 for a in self.schedule_workers.agents]
        upper = self.n_workers
        lower = -self.n_workers

        return rewards, upper, lower, n_wrongs
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """action space per agent"""
        return Tuple([
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """ obs space consisting of all agent states + adjacents matrix with edge attributes """
        graph_state = Box(0, GRAPH_HASH, shape=(1,), dtype=np.float32) # current step hash
        agent_state = [
            Discrete(2),                                                # active flag
            Discrete(3),                                                # agent type
            Discrete(self.n_oracle_states),                             # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ]
        agent_states = Tuple([Tuple(agent_state) for _ in range(self.n_agents)])

        edge_state = [
            Discrete(2), # exists flag
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32), # relative position to the given node
        ]
        edge_states = Tuple([Tuple(edge_state) for _ in range(self.n_agents * self.n_agents)])

        return Tuple([graph_state, agent_states, edge_states])
    
    def get_graph(self):
        """compute adjacency graph"""
        graph = nx.Graph()
        for worker in self.schedule_all.agents:
            graph.add_node(worker.unique_id)

        for worker in self.schedule_all.agents:
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for n in neighbors:
                graph.add_edge(worker.unique_id, n.unique_id)
        return graph
    
    def get_obss(self) -> dict:
        """ collect and return current obs"""
        step_hash = np.array([random.randint(0, GRAPH_HASH)])

        # agent states
        agent_states = [None for _ in range(self.n_agents)]
        for worker in self.schedule_all.agents:
            if type(worker) is Oracle:
                agent_states[worker.unique_id] = tuple([0,
                                         TYPE_ORACLE, 
                                         worker.state,
                                         np.zeros(self.n_hidden_states)])
            if type(worker) is Worker:
                agent_states[worker.unique_id] = tuple([0,
                                         TYPE_WORKER, 
                                         worker.output, 
                                         worker.hidden_state])
        # edge attributes
        edge_states = [None for _ in range(self.n_agents ** 2)]
        for worker in self.schedule_all.agents:
            # fully connected graph
            for destination in self.schedule_all.agents:
                rel_pos = get_relative_pos(worker.pos, destination.pos)
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = tuple([0, np.array(rel_pos, dtype=np.int32)])

            # visible graph
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for n in neighbors:
                rel_pos = get_relative_pos(worker.pos, n.pos)
                edge_states[worker.unique_id * self.n_agents + n.unique_id] = tuple([1, np.array(rel_pos, dtype=np.int32)]) 


        # make obs_space for every agent, changing the live flag
        obss = dict()
        for worker in self.schedule_workers.agents:
            curr_agent_state = agent_states.copy()
            curr_agent_state[worker.unique_id] = tuple([1,
                                         TYPE_WORKER, 
                                         worker.output, 
                                         worker.hidden_state])
            obss[worker.unique_id] = tuple([step_hash, tuple(curr_agent_state), tuple(edge_states)])
        return obss


    def step(self, actions=None) -> None:
        """advance the model one step in inference mode"""        
        # determine actions for inference mode
        if self.inference_mode:
            actions = dict()
            obss = self.get_obss()

            # @todo: sample for every agent
            if self.policy_net:
                actions = self.policy_net.compute_single_action(obss)
            else:
                for worker in self.schedule_workers.agents:
                    actions[worker.unique_id] = self.get_action_space().sample()

        # advance simulation
        for k, v in actions.items():
            worker = [w for w in self.schedule_all.agents if w.unique_id == k][0]
            worker.output = v[0]
            worker.hidden_state = v[1]
        self.ts_episode += 1
        self.ts_curr_state += 1
        self.running = self.ts_episode < self.episode_length
        
        # compute reward and state
        rewards, dupper, dlower, n_wrongs = self._compute_reward()
        self.reward_total += sum(rewards)
        self.reward_upper_bound += dupper
        self.reward_lower_bound += dlower

        rewardss = {}
        truncateds = {"__all__": self.ts_episode >= self.episode_length}
        terminateds = {"__all__": False}
        for worker in self.schedule_workers.agents:
            rewardss[worker.unique_id] = rewards[worker.unique_id - 1]

        # terminate or change to new oracle state
        old_state = self.oracle.state
        ts_old_state = self.ts_curr_state
        if self.p_oracle_change > 0 and self.running and \
            ts_old_state >= self.state_switch_pause and self.ts_episode + self.state_switch_pause <= self.episode_length:
            r = random.random()
            if r <= self.p_oracle_change:
                new_state = random.randint(0, self.n_oracle_states-1)
                while new_state == self.oracle.state:
                    new_state = random.randint(0, self.n_oracle_states-1)
                self.oracle.state = new_state
                self.n_state_switches += 1
                self.ts_curr_state = 0

        # print overview
        if self.inference_mode:
            print()
            print(f"------------- step {self.ts_episode}/{self.episode_length} ------------")
            print(f"  states             = {old_state} - {[a.output for a in self.schedule_workers.agents]}")
            print(f"  rewards            = {rewards}")
            print(f"  converged          = {n_wrongs == 0}")
            print(f"  truncated          = {self.ts_episode >= self.episode_length}")
            print()
            print(f"  next_state         = {'-' if old_state == self.oracle.state else {self.oracle.state}}")
            print(f"  state_switch_pause = {ts_old_state}/{self.state_switch_pause}")
            print(f"  n_state_switches   = {self.n_state_switches}")
            print()
            print(f"  reward_lower_bound = {self.reward_lower_bound}")
            print(f"  reward_total       = {self.reward_total}")
            print(f"  reward_upper_bound = {self.reward_upper_bound}")
            print(f"  reward_percentile  = {(self.reward_total - self.reward_lower_bound) / (self.reward_upper_bound - self.reward_lower_bound)}")
            print()

        return self.get_obss(), rewardss, terminateds, truncateds


