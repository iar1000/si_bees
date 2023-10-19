import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple, Dict

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

    def get_obs_space(self) -> gymnasium.spaces.Space:
        # calculate moore neighborhood size 

        bin_trace = Discrete(2, shape=(self.nh_size,))
        bin_plattform = Discrete(2, shape=(self.nh_size,))
        bin_plattform_occupation = Discrete(3, shape=(1,)) # -1 if not visible
        bin_oracle = Discrete(2, shape=(self.nh_size,))
        bin_oracle_directives = Discrete(3, shape=(1,)) # -1 if not visible
        comm_workers = Box(0, 1, shape=(self.nh_size, self.n_comm_vec), dtype=np.float64)
        obs_space = Tuple(
            bin_trace,
            bin_plattform,
            bin_plattform_occupation,
            bin_oracle,
            bin_oracle_directives,
            comm_workers
        )

        return obs_space
    
    def observe(self) -> dict:
        """
        return observed environment of the agent
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=self.moore_nh, radius=self.n_visibility_range)

        bin_trace = np.zeros(shape=(self.nh_size,))
        bin_plattform = np.zeros(shape=(self.nh_size,))
        bin_plattform_occupation = np.array([-1])
        bin_oracle = np.zeros(shape=(self.nh_size,))
        bin_oracle_directives = np.array([-1])
        comm_workers = np.zeros((self.nh_size, self.n_comm_vec), dtype=np.float64)
        
        for n in neighbors:
            rel_pos = get_relative_pos(self.pos, n.pos)

            if type(n) is Worker:
                comm_workers[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = n.get_comm_vec()
                # add trace
                for p in self.pos_history:
                    bin_trace[relative_moore_to_linear(get_relative_pos(self.pos, p), radius=self.n_visibility_range)] = 1
            elif type(n) is Oracle:
                bin_oracle[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = 1
                bin_oracle_directives[0] = n.get_state()
            elif type(n) is Plattform:
                bin_plattform[relative_moore_to_linear(rel_pos, radius=self.n_visibility_range)] = 1
                bin_plattform_occupation[0] = n.is_occupied()

        obs = (
            bin_trace,
            bin_plattform,
            bin_plattform_occupation,
            bin_oracle,
            bin_oracle_directives,
            comm_workers
        )

        # calculate action mask based on grid
        right_mask = np.zeros(shape=(1,), dtype=np.int8)
        left_mask = np.zeros(shape=(1,), dtype=np.int8)
        up_mask = np.zeros(shape=(1,), dtype=np.int8)
        down_mask = np.zeros(shape=(1,), dtype=np.int8)
        c_mask = np.zeros(shape=(self.n_comm_vec,), dtype=np.int8)

        x_curr, y_curr = self.pos
        if self.model.grid.out_of_bounds((x_curr + 1, y_curr)):
            right_mask[0] = 1
        if self.model.grid.out_of_bounds((x_curr - 1, y_curr)):
            left_mask[0] = 1
        if self.model.grid.out_of_bounds((x_curr, y_curr + 1)):
            up_mask[0] = 1
        if self.model.grid.out_of_bounds((x_curr, y_curr - 1)):
            down_mask[0] = 1 

        action_mask = (
            right_mask,
            left_mask,
            up_mask,
            down_mask,
            None
        )           

        observation = {"observations": obs, "action_mask": action_mask}
        return observation

    def get_action_space(self) -> gymnasium.spaces.Space:
        """
        move:   (x,y) relative direction of movement
        h:      internal state
        c:      communication output
        """
        right = Discrete(1)
        left = Discrete(1)
        up = Discrete(1)
        down = Discrete(1)
        h = Box(0, 1, shape=(self.n_hidden_vec,), dtype=np.float32)
        c = Box(0, 1, shape=(self.n_comm_vec,), dtype=np.float32)

        return Tuple(
            [right,
            left,
            up,
            down,
            # h,
            c]
        )

    def step(self, action=None) -> None:
        """
        move agent and update internal state 
        """
        # get action in inference mode
        if not action:
            obs = self.observe()
            action = self.model.policy_net.compute_single_action(obs)

        # decode action
        right, left, up, down, c = action

        # move agent
        x_curr, y_curr = self.pos
        x_updated = x_curr + right - left
        y_updated = y_curr + up - down
        pos_updated = (x_updated, y_updated)
        assert not self.model.grid.out_of_bounds(pos_updated), "action masking failed, agent out-of-bounds"

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

    def is_occupied(self) -> List[mesa.Agent]:
        return len(self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=0)) > 0


class Oracle(mesa.Agent):
    """
    oracle that displays which plattform to step on
    0 is default and means none
    """
    def __init__(self, unique_id: int, model: mesa.Model, state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.state = state

    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state