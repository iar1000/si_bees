from math import floor
import mesa
import random
import numpy as np
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete
from agents import Oracle, Worker

from utils import get_relative_pos

MAX_DISTANCE = 100

TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

class Simple_model(mesa.Model):
    """
    the oracle outputs a number which the agents need to copy
    the visibility range is limited, so the agents need to communicate the oracle output
    """

    def __init__(self, config: dict,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__()
        
        self.min_steps = config["model"]["min_steps"]
        self.max_steps = config["model"]["max_steps"]
        self.n_workers = config["model"]["n_workers"]
        self.n_agents = self.n_workers + 1
        self.n_oracle_states = config["model"]["n_oracle_states"]
        self.n_hidden_states = config["model"]["n_hidden_state"]
        self.communication_range = config["model"]["communication_range"]

        # mesa setup
        self.grid = mesa.space.MultiGrid(config["model"]["grid_width"], config["model"]["grid_height"], False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.current_id = 0
        self.curr_step = 0

        # initialisation outputs of agents
        oracle_state = random.randint(0, self.n_oracle_states-1)
        r = random.randint(0, self.n_oracle_states-1)
        while r == oracle_state:
            r = random.randint(0, self.n_oracle_states-1)
        worker_output = r

        # place agents
        self.oracle = Oracle(self._next_id(), self, state=oracle_state)
        self.grid.place_agent(agent=self.oracle, pos=(floor(config["model"]["grid_width"] / 2), floor(config["model"]["grid_height"] / 2)))
        self.schedule.add(self.oracle)
        for _ in range(self.n_workers):
            x_old, y_old = self.schedule.agents[-1].pos
            x_new, y_new = x_old + self.communication_range - 1, y_old
            worker = Worker(self._next_id(), self, output=worker_output, n_hidden_states=self.n_hidden_states)
            self.grid.place_agent(agent=worker, pos=(x_new, y_new))
            self.schedule.add(worker)

        # inference mode
        self.inference_mode = inference_mode
        self.policy_net = policy_net

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
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
                print("prick")
        
        # proceed simulation
        for i, worker in enumerate(self.schedule.agents[1:]):
            assert type(worker) == Worker
            worker.output = actions[i][0]
            worker.hidden_state = actions[i][1]
        self.curr_step += 1
        
        # compute reward and state
        wrongs = sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])
        reward = 10 if wrongs == 0 else -wrongs
        terminated = wrongs == 0 and self.curr_step >= self.min_steps
        truncated = self.curr_step >= self.max_steps

        # print overview
        if self.inference_mode:
            print(f"step {self.curr_step}")
            print(f"  oracle_state    = {self.oracle.state}")
            print(f"  worker_outputs  = {[a.output for a in self.schedule.agents if type(a) is Worker]}")
            print(f"  reward          = {reward}")
            print(f"  terminated      = {terminated}")
            

        return self.get_obs(), reward, terminated, truncated



    

