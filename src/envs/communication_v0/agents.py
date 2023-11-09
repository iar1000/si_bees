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
                 size_com_vec: int = 8, com_range: int = 4,
                 len_trace: int = 1):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"

        # neighborhood variables
        self.com_range = com_range
        self.moore_nh = True
        self.nh_size = (2 * self.com_range + 1)**2

        # internal state
        self.size_com_vec = size_com_vec
        self.comm_vec = np.zeros(shape=(size_com_vec,))
        
        # trace
        self.len_trace = len_trace
        self.trace_global_locations = list()


    def get_comm_vec(self) -> np.array:
        return self.comm_vec

    def get_obs_space(self) -> dict:
        """
        trace: neihbourhood with past positions
        plattform_location: one_hot encoded location in neighbourhood
        plattform_occupation: -1 if not visible, else 0/1 if it is occupied
        oracle_loaction: one_hot encoded location in neighbourhood
        oracle_state: -1 if not visible, else what the oracle is saying
        agent_nh: visible states of all visible neighbours
        """
        trace = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        plattform_location = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        plattform_occupation = Box(-1, 1, shape=(1,), dtype=np.int8)
        oracle_location = Box(0, 1, shape=(self.nh_size,), dtype=np.int8)
        oracle_state = Box(-1, 1, shape=(1,), dtype=np.int8)
        agent_nh = [Box(0, 1, shape=(self.size_com_vec, ), dtype=np.float64) for _ in range(self.nh_size)]
        
        spaces = [trace, plattform_location, plattform_occupation, oracle_location, oracle_state] + agent_nh

        return Tuple(spaces)
    
    def observe(self) -> dict:
        """see get_obs_space"""
        # set trace
        trace = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        for p_trace in self.trace_global_locations:
            trace[relative_moore_to_linear(get_relative_pos(self.pos, p_trace), radius=self.com_range)] = 1
            
        # observations
        plattform_location = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        plattform_occupation = np.array([-1], dtype=np.int8)
        oracle_location = np.zeros(shape=(self.nh_size,), dtype=np.int8)
        oracle_state = np.array([-1], dtype=np.int8)
        agent_nh = [np.zeros((self.size_com_vec,), dtype=np.float64) for _ in range(self.nh_size)]
        

        neighbors = self.model.grid.get_neighbors(self.pos, moore=self.moore_nh, radius=self.com_range)
        for n in neighbors:
            rel_pos = get_relative_pos(self.pos, n.pos)

            if type(n) is Worker:
                agent_nh[relative_moore_to_linear(rel_pos, radius=self.com_range)] = n.get_comm_vec()
            elif type(n) is Oracle:
                oracle_location[relative_moore_to_linear(rel_pos, radius=self.com_range)] = 1
                oracle_state[0] = n.get_state()
            elif type(n) is Plattform:
                plattform_location[relative_moore_to_linear(rel_pos, radius=self.com_range)] = 1
                plattform_occupation[0] = 1 if n.is_occupied() else 0

        obss = [trace, plattform_location, plattform_occupation, oracle_location, oracle_state] + agent_nh
        
        return tuple(obss)

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
        c:      communication output
        """
        move_x = Discrete(3)
        move_y = Discrete(3)
        c = Box(0, 1, shape=(self.size_com_vec,), dtype=np.float64)

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

        # update comm-vector
        self.comm_vec = c

        # move agent within bounds, ignore out of bounds movement
        old_pos = self.pos
        x_curr, y_curr = self.pos
        x_new = x_curr+1 if move_x == 1 else (x_curr-1 if move_x == 2 else x_curr)
        y_new = y_curr+1 if move_y == 1 else (y_curr-1 if move_y == 2 else y_curr)
        x_updated = x_curr if self.model.grid.out_of_bounds((x_new, y_curr)) else x_new
        y_updated = y_curr if self.model.grid.out_of_bounds((x_curr, y_new)) else y_new

        pos_updated = (x_updated, y_updated)
        self.model.grid.move_agent(self, pos_updated)
        
        # update current location to trace
        self.trace_global_locations.append(old_pos)
        if len(self.trace_global_locations) > self.len_trace:
            self.trace_global_locations.pop(0)
        assert len(self.trace_global_locations) <= self.len_trace, "position history should not be longer than maximum allowed trace length"


class Plattform(mesa.Agent):
    """
    plattform that can be stepped on
    """
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
        self.name = f"plattform_{unique_id}"

    def is_occupied(self) -> bool:
        nbs = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=0)
        nbs = [nb for nb in nbs if type(nb) is not Plattform]
        return len(nbs) > 0


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