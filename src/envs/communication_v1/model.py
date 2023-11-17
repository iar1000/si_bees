import mesa
from math import floor
import numpy as np
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple
from gymnasium.spaces.utils import flatten_space

from utils import get_random_pos_on_border, get_relative_pos
from envs.communication_v1.agents import Oracle, Plattform, Worker 


class CommunicationV1_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self,
                 max_steps: int,
                 n_agents: int, agent_placement: str,
                 plattform_distance: int, oracle_burn_in: int, p_oracle_change: float,
                 n_tiles_x: int, n_tiles_y: int,
                 size_com_vec: int, com_range: int, len_trace: int,
                 policy_net: Algorithm = None) -> None:
        super().__init__()

        self.policy_net = policy_net # not None in inference mode

        self.n_agents = n_agents
        self.com_range = com_range
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y

        self.max_steps = max_steps
        self.n_steps = 0 # current number of steps
        self.oracle_burn_in = oracle_burn_in
        self.p_oracle_change = p_oracle_change

        assert n_agents > 0, "must have a positive number of agents"
        assert agent_placement in ["center", "random"], f"agent placement {agent_placement} unknown"
        assert n_tiles_x > 2 and n_tiles_y > 2, "minimum map size of 3x3 required"
        assert 0 < plattform_distance, "distance of plattform to oracle must be at least 1"

        # create model
        # grid coordinates, bottom left = (0,0)
        self.grid = mesa.space.MultiGrid(n_tiles_x, n_tiles_y, False)
        self.schedule = mesa.time.BaseScheduler(self)

        # map centerpoint
        y_mid = floor(n_tiles_y / 2)
        x_mid = floor(n_tiles_x / 2)
        assert x_mid >= plattform_distance and y_mid >= plattform_distance, "plattform distance to oracle is too large, placement will be out-of-bounds"

        # create workers
        for _ in range(n_agents):
            new_worker = Worker(self._next_id(), self, 
                                size_com_vec=size_com_vec,
                                com_range=com_range,
                                len_trace=len_trace)
            self.schedule.add(new_worker)
            self.grid.place_agent(agent=new_worker, pos=(x_mid, y_mid))
            if agent_placement == "random":
                self.grid.move_to_empty(agent=new_worker)

        # place oracle in the middle and lightswitch around it
        self.oracle = Oracle(self._next_id(), self)
        self.plattform = Plattform(self._next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_mid, y_mid))
        self.grid.place_agent(agent=self.plattform, pos=get_random_pos_on_border(center=(x_mid, y_mid), dist=plattform_distance))

        # track reward, max reward is the optimal case
        self.accumulated_reward = 0
        self.last_reward = 0
        self.max_reward = 0
        self.reward_delay = int(floor(plattform_distance / com_range)) + 1
        self.time_to_reward = 0

        # sizes for later processing, must be updated if obs_space change
        self.agent_obs_size = 6
        self.adj_matrix_size = self.n_agents ** 2
        self.obs_space_size = self.n_agents * self.agent_obs_size + self.adj_matrix_size
        self.agent_action_size = 2

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def print_status(self) -> None:
        """print status of the model"""
        print(f"step {self.n_steps}: oracle is {'deactived' if not self.oracle.is_active() else self.oracle.get_state()}\n\ttime to reward={self.time_to_reward}\n\treward={self.last_reward}, acc_reward={self.accumulated_reward}/{self.max_reward}")

    def print_agent_locations(self) -> None:
        """print a string with agent locations"""
        oracle_state = self.oracle.get_state()
        out = f"step {self.n_steps}; o={oracle_state}, "
        for agent in self.schedule.agents:
            out += f"{agent.name}: {agent.pos} "
        print(out)
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """action spaces of all agents"""
        move = Box(-1, 1, shape=(2,), dtype=np.int8) # relative movement in x and y direction

        return flatten_space(Tuple([move for _ in range(self.n_agents)]))
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """obs space consisting of all agent states + adjacents matrix"""
        plattform_location = Box(-self.com_range, self.com_range, shape=(2,)) # relative position of plattform
        oracle_location = Box(-self.com_range, self.com_range, shape=(2,)) # relative position of oracle
        plattform_occupation = Box(-1, 1, shape=(1,)) # -1 if not visible, else 0/1 if it is occupied
        oracle_state = Box(-1, 1, shape=(1,)) # -1 if not visible, else what the oracle is saying
        agent_state = flatten_space(Tuple([plattform_location, oracle_location, plattform_occupation, oracle_state]))
        all_agent_states = flatten_space(Tuple([agent_state for _ in range(self.n_agents)]))

        adj_matrix = Box(0, 1, shape=(self.n_agents * self.n_agents,), dtype=np.int8)        
        flat_obs = flatten_space(Tuple([all_agent_states, adj_matrix]))

        print("\n=== obs space ===")
        print(f"agent_state     : {agent_state}")
        print(f"adj matrix size : {self.adj_matrix_size}")
        print(f"total size      : {self.obs_space_size}")

        return flat_obs
    
    def get_obs(self) -> dict:
        """
        gather information about all agents states and their connectivity.
        fill the observation in the linear obs_space with the same format as described in get_obs_space
        """
        # bugfix, somehow get_obs() get's called before get_obs_space()
        if not self.obs_space_size:
            self.obs_space_size = self.get_obs_space().shape[0]

        obs = np.zeros(shape=(self.obs_space_size,))
        adj_matrix_offset = self.n_agents * self.agent_obs_size
        for worker in self.schedule.agents:
            obs_offset = worker.unique_id * self.agent_obs_size 
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.com_range, include_center=True)
            for n in neighbors:
                rel_pos = get_relative_pos(worker.pos, n.pos)
                if type(n) is Plattform:
                    obs[obs_offset], obs[obs_offset + 1] = rel_pos
                    obs[obs_offset + 4] = 1 if n.is_occupied() else 0
                elif type(n) is Oracle:
                    obs[obs_offset + 2], obs[obs_offset + 3] = rel_pos
                    obs[obs_offset + 5] = n.get_state()
                # adj. matrix
                elif type(n) is Worker and n is not worker:
                    obs[adj_matrix_offset + n.unique_id * self.n_agents + worker.unique_id] = 1
                    obs[adj_matrix_offset + worker.unique_id * self.n_agents + n.unique_id] = 1

        return obs
        
    def apply_actions(self, actions) -> None:
        """apply the actions to the indivdual agents"""
        for i, worker in enumerate(self.schedule.agents):
            x_old, y_old = worker.pos
            x_new = max(0, min(self.n_tiles_x - 1, x_old + actions[i * self.agent_action_size]))
            y_new = max(0, min(self.n_tiles_y - 1, y_old + actions[i * self.agent_action_size + 1]))
            self.grid.move_agent(worker, (x_new, y_new))
        
    def finish_round(self) -> [int, bool]:
        """
        finish up a round
        - increases the round counter by 1
        - change oracle state
        - count points
        """
        # update round
        self.n_steps += 1
        last_reward_is, last_reward_could = self.compute_reward()
        self.last_reward = last_reward_is
        self.accumulated_reward += last_reward_is
        self.max_reward += last_reward_could

        # activate oracle
        if not self.oracle.is_active() and self.n_steps >= self.oracle_burn_in:
            self.oracle.activate()
            self.oracle.set_state(1)
            self.time_to_reward = self.reward_delay
        # switch oracle state with certain probability
        elif self.oracle.is_active() and self.time_to_reward == 0:
            r = self.random.random()
            if r < self.p_oracle_change:
                curr_state = self.oracle.get_state()
                self.oracle.set_state((curr_state + 1) % 2)
                self.time_to_reward = self.reward_delay
        else:
            self.time_to_reward = max(0, self.time_to_reward - 1)
        
        return self.last_reward, self.max_steps <= self.n_steps
    
    def compute_reward(self) -> [int, int]:
        """computes the reward based on the current state and the reward that could be achieved in the optimal case"""
        oracle_state = self.oracle.get_state()
        plattform_occupation = self.plattform.is_occupied()

        # dont go on plattform if oracle is not active
        if not self.oracle.is_active():
            if plattform_occupation == 1:
                return -1, 0
            else:
                return 0, 0
        else:
            # time delay to diffuse oracle instruction to all agents
            if self.time_to_reward > 0:
                return 0, 0
            elif oracle_state == 1:
                if plattform_occupation == 1:
                    return 1, 1
                else:
                    return -1, 1
            elif oracle_state == 0:
                if plattform_occupation == 1:
                    return -1, 0
                else:
                    return 0, 0

    

