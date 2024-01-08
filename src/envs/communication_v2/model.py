import os
import random
import mesa
from math import floor
import numpy as np
from ray.rllib.algorithms import Algorithm
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from matplotlib import pyplot as plt

import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete

from utils import get_random_pos_on_border, get_relative_pos
from envs.communication_v2.agents import Oracle, Worker 

MAX_DISTANCE = 100

TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

class CommunicationV2_model(mesa.Model):
    """
    the oracle outputs a number which the agents need to copy
    the visibility range is limited, so the agents need to communicate the oracle output
    """

    def __init__(self,
                 max_steps: int,
                 n_workers: int | str,
                 n_oracle_states: int, p_oracle_change: float,
                 n_tiles_x: int, n_tiles_y: int,
                 size_hidden_vec: int, com_range: int,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__()
        
        # simulation t
        self.max_steps = max_steps
        self.curr_step = 0
        assert max_steps > 0, "max_steps must be positive"

        # grid  
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        assert 0 < self.n_tiles_x < MAX_DISTANCE and 0 < self.n_tiles_y < MAX_DISTANCE , f"grid size not in bounds {MAX_DISTANCE}"

        # worker specs
        self.n_workers = n_workers
        self.n_agents = n_workers + 1
        self.n_oracle_state = n_oracle_states
        self.size_hidden_vec = size_hidden_vec
        self.com_range = com_range

        # mesa setup
        self.grid = mesa.space.MultiGrid(n_tiles_x, n_tiles_y, False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.current_id = 0

        # place agents
        self.oracle = Oracle(self._next_id(), self, state=0)
        self.grid.place_agent(agent=self.oracle, pos=(floor(n_tiles_x / 2), floor(n_tiles_y / 2)))
        self.schedule.add(self.oracle)
        for _ in range(n_workers):
            x_old, y_old = self.schedule.agents[-1].pos
            x_new, y_new = x_old + com_range - 1, y_old
            worker = Worker(self._next_id(), self,
                            hidden_vec=np.zeros(shape=(size_hidden_vec,)),
                            output=n_oracle_states - 1)
            self.grid.place_agent(agent=worker, pos=(x_new, y_new))
            self.schedule.add(worker)
        assert len(self.schedule.agents) == self.n_agents, "didn't add all workers to the schedule"

        # inference mode @todo
        if inference_mode:
            self.policy_net = policy_net
            self.datacollector = mesa.DataCollector(model_reporters={
                "max_total_reward": lambda x: self.max_total_reward,
                "accumulated_reward": lambda x: self.accumulated_reward,
                "max_obtainable_reward": lambda x: self.max_obtainable_reward,
                "accumulated_obtainable_reward": lambda x: self.accumulated_obtainable_reward,
                "last_reward": lambda x: self.last_reward,
                "score": lambda x: max(0, self.accumulated_obtainable_reward) / self.max_obtainable_reward * 100 if self.max_obtainable_reward > 0 else 0,
                }
            )

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def save_graph(self, path: str) -> None:
        """save the current graph to a file"""
        # build graph to plot
        obs = self.get_obs()
        edges = obs[1]
        actor_froms, actor_tos, actor_edge_attr = [], [], []
        fc_froms, fc_tos, fc_edge_attr = [], [], []
        for j in range(self.n_agents ** 2):
            if not j // self.n_agents == j % self.n_agents: # no self loops
                if edges[j][0] == 1:
                    actor_froms.append(j // self.n_agents)
                    actor_tos.append(j % self.n_agents)
                    actor_edge_attr.append(edges[j][1:])

                # add edge to fc graph
                fc_froms.append(j // self.n_agents)
                fc_tos.append(j % self.n_agents)
                fc_edge_attr.append(edges[j][1:])

        # build edge indexes
        actor_edge_index = torch.tensor([actor_froms, actor_tos], dtype=torch.int64)
        fc_edge_index = torch.tensor([fc_froms, fc_tos], dtype=torch.int64)

        # plot graph
        plt.figure(figsize=(10, 4))
        data = Data(edge_index=actor_edge_index, num_nodes=self.n_agents)
        g = to_networkx(data)
        pos = nx.kamada_kawai_layout(g)  
        nx.draw_networkx_edges(g, pos,
                               connectionstyle='arc3, rad = 0')
        nx.draw_networkx_nodes(g, pos, 
                               node_size=100,
                               node_color=['green'] + ['black'] * (self.n_workers))
        nx.draw_networkx_labels(g, pos, font_size=8, font_color="white")
        nx.draw_networkx_edge_labels(g, pos, 
                                     {e: actor_edge_attr[i][0].tolist() for i, e in enumerate(g.edges)},
                                     label_pos=0.75,
                                     font_size=7)
        plt.savefig(fname=path)
    
    def print_status(self) -> None:
        """print status of the model"""
        print(f"step {self.curr_step}")
        print(f"  oracle_state    = {self.oracle.state}")
        print(f"  worker_outputs  = {[a.output for a in self.schedule.agents if type(a) is Worker]}")
        print(f"  reward          = {-sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])}")
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """
        hidden vector is used as input for the gnn
        output must be same as oracle state
        """
        agent_actions = [
            Box(0, 1, shape=(self.size_hidden_vec,), dtype=np.float32), # hidden vector
            Discrete(self.n_oracle_state), # output
        ]
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """ obs space consisting of all agent states + adjacents matrix with edge attributes """
        agent_state = [
            Discrete(3), # agent type
            Box(0, 1, shape=(self.size_hidden_vec,), dtype=np.float32) # hidden vector
        ]
        agent_states = Tuple([Tuple(agent_state) for _ in range(self.n_agents)])

        edge_state = [
            Discrete(2), # exists flag
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.int32), # relative position to the given node
        ]
        edge_states = Tuple([Tuple(edge_state) for _ in range(self.n_agents * self.n_agents)])

        return Tuple([agent_states, edge_states])
    
    def get_obs(self) -> dict:
        """ collect and return current obs"""
        agent_states = [None for _ in range(self.n_agents)]
        for i, worker in enumerate(self.schedule.agents):
            if type(worker) is Oracle:
                agent_states[i] = tuple([TYPE_ORACLE, 
                                         np.full(self.size_hidden_vec, worker.state / 10, dtype=np.float32)])
            if type(worker) is Worker:
                agent_states[i] = tuple([TYPE_WORKER, 
                                         worker.hidden_vec])

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

        assert all([x is not None for x in agent_states]), "not all agent states set"
        assert all([x is not None for x in edge_states]), "not all edges set"
        return tuple([tuple(agent_states), tuple(edge_states)])

    def step(self, actions=None) -> None:
        """advance the model one step in inference mode"""        
        # determine actions for inference mode
        if not actions:
            if self.policy_net is None:
                actions = self.get_action_space().sample()
            else:
                actions = self.policy_net.compute_single_action(self.get_obs())
        
        # proceed simulation
        for i, worker in enumerate(self.schedule.agents[1:]):
            assert type(worker) == Worker
            worker.hidden_vec = actions[i][0]
            worker.output = actions[i][1]
        new_obs = self.get_obs()

        if not actions:
            self.print_status()
        
        # compute reward and state
        reward = -sum([1 for a in self.schedule.agents if type(a) is Worker and a.output != self.oracle.state])
        terminated = reward == 0
        truncated = self.curr_step > self.max_steps
        self.curr_step += 1
        # collect data
        #self.datacollector.collect(self)

        return new_obs, reward, terminated, truncated



    

