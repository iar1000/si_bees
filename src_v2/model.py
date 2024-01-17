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
        
        # simulation t
        self.max_steps = 20
        self.curr_step = 0

        self.n_workers = 1
        self.n_agents = self.n_workers + 1
        self.n_oracle_states = 5
        self.com_range = 2

        # mesa setup
        self.grid = mesa.space.MultiGrid(21, 21, False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.current_id = 0

        # initialisation outputs of agents
        oracle_state = random.randint(0, self.n_oracle_states-1)
        r = random.randint(0, self.n_oracle_states-1)
        while r == oracle_state:
            r = random.randint(0, self.n_oracle_states-1)
        worker_output = r

        # place agents
        self.oracle = Oracle(self._next_id(), self, state=oracle_state)
        self.grid.place_agent(agent=self.oracle, pos=(11, 11))
        self.schedule.add(self.oracle)
        for _ in range(self.n_workers):
            x_old, y_old = self.schedule.agents[-1].pos
            x_new, y_new = x_old + self.com_range - 1, y_old
            worker = Worker(self._next_id(), self, output=worker_output)
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
        """
        hidden vector is used as input for the gnn
        output must be same as oracle state
        """
        agent_actions = [
            Discrete(self.n_oracle_states), # output
        ]
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """ obs space consisting of all agent states + adjacents matrix with edge attributes """
        agent_state = [
            Discrete(3), # agent type
            #Box(0, 1, shape=(self.size_hidden_vec,), dtype=np.float32), # hidden vector
            Discrete(self.n_oracle_states) # current output
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
                                         #np.zeros(self.size_hidden_vec),
                                         worker.state])
            if type(worker) is Worker:
                agent_states[i] = tuple([TYPE_WORKER, 
                                         #worker.hidden_vec, 
                                         worker.output])
        # edge attributes
        edge_states = [None for _ in range(self.n_agents ** 2)]
        for i, worker in enumerate(self.schedule.agents):
            # fully connected graph
            for j, destination in enumerate(self.schedule.agents):
                rel_pos = get_relative_pos(worker.pos, destination.pos)
                edge_states[i * self.n_agents + j] = tuple([0, np.array(rel_pos, dtype=np.int32)])

            # visible graph
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.com_range, include_center=True)
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
                print("prick")
        
        # proceed simulation
        for i, worker in enumerate(self.schedule.agents[1:]):
            assert type(worker) == Worker
            worker.output = actions[i][0]
        self.curr_step += 1
        
        # compute reward and state
        wrongs = sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])
        reward = 10 if wrongs == 0 else -wrongs
        terminated = wrongs == 0 and self.curr_step > 7
        truncated = self.curr_step > self.max_steps

        # print overview
        if self.inference_mode:
            print(f"step {self.curr_step}")
            print(f"  oracle_state    = {self.oracle.state}")
            print(f"  worker_outputs  = {[a.output for a in self.schedule.agents if type(a) is Worker]}")
            print(f"  reward          = {reward}")
            print(f"  terminated      = {terminated}")
            

        return self.get_obs(), reward, terminated, truncated



    

