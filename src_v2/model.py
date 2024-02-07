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

class Simple_model(mesa.Model):
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
        self.schedule = mesa.time.BaseScheduler(self)
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
        self.schedule.add(self.oracle)
        agent_positions = compute_agent_placement(self.n_workers, self.communication_range, 
                                                  self.grid_size, self.grid_size, 
                                                  oracle_pos, self.worker_placement)
        for i, curr_pos in enumerate(agent_positions):
            worker = Worker(self._next_id(), self, output=worker_output[i], n_hidden_states=self.n_hidden_states)
            self.grid.place_agent(agent=worker, pos=curr_pos)
            self.schedule.add(worker)

        # inference mode
        self.inference_mode = inference_mode
        self.policy_net = policy_net

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def _compute_reward(self) -> [int, int, int, int]:
        assert self.reward_calculation in {"individual", "binary"}

        # compute reward
        REWARD = 10
        n_wrongs = sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])
        reward = REWARD if n_wrongs == 0 else -n_wrongs if self.reward_calculation == "individual" else -REWARD
        upper = REWARD
        lower = -self.n_workers if self.reward_calculation == "individual" else -REWARD

        return reward, upper, lower, n_wrongs
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        agent_actions = [
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ]
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """ obs space consisting of all agent states + adjacents matrix with edge attributes """
        agent_state = [
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

        return Tuple([agent_states, edge_states])
    
    def get_graph(self):
        """compute adjacency graph"""
        graph = nx.Graph()
        for i, worker in enumerate(self.schedule.agents):
            graph.add_node(i)
        for i, worker in enumerate(self.schedule.agents):
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for n in neighbors:
                graph.add_edge(i, n.unique_id)
        return graph
    
    def get_obs(self) -> dict:
        """ collect and return current obs"""
        agent_states = [None for _ in range(self.n_agents)]
        for i, worker in enumerate(self.schedule.agents):
            if type(worker) is Oracle:
                agent_states[i] = tuple([TYPE_ORACLE, 
                                         worker.state,
                                         np.zeros(self.n_hidden_states)])
            if type(worker) is Worker:
                agent_states[i] = tuple([TYPE_WORKER, 
                                         worker.output, 
                                         worker.hidden_state])
        # edge attributes
        edge_states = [None for _ in range(self.n_agents ** 2)]
        for i, worker in enumerate(self.schedule.agents):
            # fully connected graph
            for j, destination in enumerate(self.schedule.agents):
                rel_pos = get_relative_pos(worker.pos, destination.pos)
                edge_states[i * self.n_agents + j] = tuple([0, np.array(rel_pos, dtype=np.int32)])

            # visible graph
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for n in neighbors:
                rel_pos = get_relative_pos(worker.pos, n.pos)
                edge_states[i * self.n_agents + n.unique_id] = tuple([1, np.array(rel_pos, dtype=np.int32)]) 

        return tuple([tuple(agent_states), tuple(edge_states)])


    def step(self, actions=None) -> None:
        """advance the model one step in inference mode"""        
        # determine actions for inference mode
        if self.inference_mode:
            if self.policy_net:
                actions = self.policy_net.compute_single_action(self.get_obs())
            else:
                actions = self.get_action_space().sample()
        
        # advance simulation
        for i, worker in enumerate(self.schedule.agents[1:]):
            assert type(worker) == Worker
            worker.output = actions[i][0]
            worker.hidden_state = actions[i][1]
            if self.model_type == MODEL_TYPE_MOVING:
                dx, dy = actions[i][2]
                dx = -1 if dx <= -0.3 else 1 if dx >= 0.3 else 0
                dy = -1 if dy <= -0.3 else 1 if dy >= 0.3 else 0
                x = max(0, min(self.grid_size-1, worker.pos[0] + dx))
                y = max(0, min(self.grid_size-1, worker.pos[1] + dy))
                self.grid.move_agent(agent=worker, pos=(x,y))
        self.ts_episode += 1
        self.ts_curr_state += 1
        
        # compute reward and state
        reward, dupper, dlower, n_wrongs = self._compute_reward()
        self.reward_total += reward
        self.reward_upper_bound += dupper
        self.reward_lower_bound += dlower

        truncated = self.ts_episode >= self.episode_length
        terminated = False
        self.running = not truncated

        # terminate or change to new oracle state
        old_state = self.oracle.state
        ts_old_state = self.ts_curr_state
        if self.p_oracle_change > 0 and not truncated and \
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
            print(f"  states             = {old_state} - {[a.output for a in self.schedule.agents if type(a) is Worker]}")
            print(f"  reward             = {reward}")
            print(f"  converged          = {n_wrongs == 0}")
            print(f"  truncated          = {truncated}")
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

        return self.get_obs(), reward, terminated, truncated
    

class Moving_model(Simple_model):
    """
    the oracle outputs a number which the agents need to copy
    the visibility range is limited, so the agents need to communicate the oracle output
    agents are capable of moving arounud on the field
    reward: how many agents have the correct output + distance
    """

    def __init__(self, config: dict,
                 use_cuda: bool = False,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__(config=config,
                         model_type=MODEL_TYPE_MOVING, 
                         use_cuda=use_cuda, 
                         policy_net=policy_net, inference_mode=inference_mode)

    def _compute_reward(self) -> [int, int, int, int]:
        assert self.reward_calculation in {"distance", "connectivity-spread"}

        # compute reward
        n_wrongs = sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])        
        reward = 0
        upper = 0
        lower = -self.n_workers
        if n_wrongs == 0:
            if self.reward_calculation == "distance":
                workers = [a for a in self.schedule.agents if type(a) is Worker]
                for w in workers:
                    dx, dy = get_relative_pos(w.pos, self.oracle.pos)
                    reward += max(abs(dx), abs(dy))
            elif self.reward_calculation == "connectivity-spread":
                g = self.get_graph()
                for i, agent in enumerate(self.schedule.agents):
                    if type(agent) is Worker:
                        dx, dy = get_relative_pos(agent.pos, self.oracle.pos)
                        if nx.has_path(g, 0, i):
                            reward += max(abs(dx), abs(dy))
        else:
            reward = -n_wrongs

        # punish completely disconected components
        if reward == 0:
            reward = -1

        # upper bound
        for i in range(self.n_workers):
            upper += min((i+1) * self.communication_range, self.grid_middle)

        return reward, upper, lower, n_wrongs
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        agent_actions = [
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
            Box(-1, 1, shape=(2,), dtype=np.float32),                   # movement x,y
        ]
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])

