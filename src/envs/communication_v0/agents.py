import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple, Dict
from gymnasium.spaces.utils import flatten_space, flatdim

import numpy as np
import mesa
from typing import List

from utils import get_relative_pos, relative_moore_to_linear
    
class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model,
                 n_hidden_vec: int = 8,
                 n_comm_vec: int = 8,
                 n_visibility_range: int = 4,
                 n_comm_range: int = 4,
                 n_trace_length: int = 1):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"

        # neighborhood variables
        self.n_visibility_range = n_visibility_range
        self.moore_nh = True
        self.nh_size = (2 * self.n_visibility_range + 1)**2

        # internal state
        self.n_comm_range = n_comm_range
        self.n_comm_vec = n_comm_vec
        self.comm_vec = np.zeros(shape=(n_comm_vec,))
        
        self.n_hidden_vec = n_hidden_vec
        self.hidden_vec = np.zeros(shape=(n_hidden_vec,))
        
        self.n_trace_length = n_trace_length
        self.pos_history = list()


    def get_comm_vec(self) -> np.array:
        return self.comm_vec

    def get_obs_space(self) -> dict:
        # obs space
        nh_trace = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        nh_plattform = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        bin_plattform_occupation = Box(-1, 1, shape=(1,), dtype=np.int8)
        nh_oracle = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        bin_oracle_directives = Box(-1, 1, shape=(1,), dtype=np.int8)
        comm_workers = Box(0, 1, shape=(self.nh_size, self.n_comm_vec), dtype=np.float64)
        obs_space = Tuple(spaces=[
            nh_trace,
            nh_plattform,
            bin_plattform_occupation,
            nh_oracle,
            bin_oracle_directives,
            #comm_workers
        ])

        return obs_space
    
    def observe(self) -> dict:
        """
        return observed environment of the agent
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=self.moore_nh, radius=self.n_visibility_range)

        nh_trace = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        nh_plattform = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        bin_plattform_occupation = np.array([-1], dtype=np.int8)
        nh_oracle = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        bin_oracle_directives = np.array([-1], dtype=np.int8)
        comm_workers = np.zeros((self.nh_size, self.n_comm_vec), dtype=np.float64)
        
        for n in neighbors:
            rel_pos = get_relative_pos(self.pos, n.pos)

            if type(n) is Worker:
                comm_workers[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = n.get_comm_vec()
                # add trace
                for p in self.pos_history:
                    nh_trace[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = 1
            elif type(n) is Oracle:
                nh_oracle[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = 1
                bin_oracle_directives[0] = n.get_state()
            elif type(n) is Plattform:
                nh_plattform[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = 1
                bin_plattform_occupation[0] = len(n.get_occupants()) > 0

        obs = tuple([
            nh_trace,
            nh_plattform,
            bin_plattform_occupation,
            nh_oracle,
            bin_oracle_directives,
            #comm_workers
        ])
        
        return obs

    def get_action_space(self) -> gymnasium.spaces.Space:
        """
        move_x:
            0: idle
            1: right
            2: left
        move_y:
            0: idle
            1: up
            2: down
        h:      internal state
        c:      communication output
        """
        move_x = Discrete(3)
        move_y = Discrete(3)
        h = Box(0, 1, shape=(self.n_hidden_vec,), dtype=np.float64)
        c = Box(0, 1, shape=(self.n_comm_vec,), dtype=np.float64)

        action_space = Tuple([move_x, move_y, c])

        return action_space

    def step(self, action=None) -> None:
        """
        move agent and update internal state 
        """
        # get action in inference mode
        if not action:
            obs = self.observe()
            # sample from given policy
            if self.model.has_policy():
                action = self.model.policy_net.compute_single_action(obs)
            # random sampling
            else:
                action = self.get_action_space().sample()

        # decode action
        move_x, move_y, c = action

        # move agent within bounds, ignore out of bounds movement
        x_curr, y_curr = self.pos
        x_new = x_curr+1 if move_x == 1 else (x_curr-1 if move_x == 2 else x_curr)
        y_new = y_curr+1 if move_y == 1 else (y_curr-1 if move_y == 2 else y_curr)
        x_updated = x_curr if self.model.grid.out_of_bounds((x_new, y_curr)) else x_new
        y_updated = y_curr if self.model.grid.out_of_bounds((x_curr, y_new)) else y_new
        print(f"agent {self.name}: {self.pos} -> {(x_updated, y_updated)}")

        pos_updated = (x_updated, y_updated)
        self.model.grid.move_agent(self, pos_updated)
        
        # update position history
        self.pos_history.append(pos_updated)
        if len(self.pos_history) > self.n_trace_length:
            self.pos_history.pop(0)
        assert len(self.pos_history) <= self.n_trace_length, "position history should not be longer than maximum allowed trace length"
    

class Plattform(mesa.Agent):
    """
    plattform that can be stepped on
    """
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
        self.name = f"plattform_{unique_id}"

    def get_occupants(self) -> bool:
        nbs = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=0)
        nbs = [nb for nb in nbs if type(nb) is not Plattform]
        return nbs


class Oracle(mesa.Agent):
    """
    oracle that displays which plattform to step on
    """
    def __init__(self, unique_id: int, model: mesa.Model, 
                 is_active: bool = False, 
                 state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.active = is_active
        self.state = state

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def is_active(self):
        return self.active
    
    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state