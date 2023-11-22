import random
import mesa
from math import floor
import numpy as np
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple
from gymnasium.spaces.utils import flatten_space

from utils import get_random_pos_on_border, get_relative_pos
from envs.communication_v1.agents import Oracle, Platform, Worker 


class CommunicationV1_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self,
                 max_steps: int,
                 n_agents: int, agent_placement: str,
                 platform_distance: int, oracle_burn_in: int, p_oracle_change: float,
                 n_tiles_x: int, n_tiles_y: int,
                 size_com_vec: int, com_range: int, len_trace: int,
                 policy_net: Algorithm = None,
                 platform_placement: str = None) -> None:
        super().__init__()

        self.policy_net = policy_net # not None in inference mode

        self.n_agents = n_agents
        self.size_com_vec = size_com_vec
        self.com_range = com_range
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y

        self.max_steps = max_steps
        self.n_steps = 0 # current number of steps
        self.oracle_burn_in = oracle_burn_in
        self.p_oracle_change = p_oracle_change

        # grid coordinates, bottom left = (0,0)
        self.grid = mesa.space.MultiGrid(n_tiles_x, n_tiles_y, False)
        self.schedule = mesa.time.BaseScheduler(self)

        # map centerpoint
        y_mid = floor(n_tiles_y / 2)
        x_mid = floor(n_tiles_x / 2)
        assert x_mid >= platform_distance and y_mid >= platform_distance, "platform distance to oracle is too large, placement will be out-of-bounds"

        # create workers
        for _ in range(n_agents):
            new_worker = Worker(self._next_id(), self, comm_vec=np.random.rand(size_com_vec))
            self.schedule.add(new_worker)
            self.grid.place_agent(agent=new_worker, pos=(x_mid, y_mid))
            if agent_placement == "random":
                self.grid.move_to_empty(agent=new_worker)

        # place oracle in the middle and lightswitch around it
        self.oracle = Oracle(self._next_id(), self)
        self.platform = Platform(self._next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_mid, y_mid))
        platform_distance = platform_distance if platform_placement is None else random.randint(1, platform_distance)
        self.grid.place_agent(agent=self.platform, pos=get_random_pos_on_border(center=(x_mid, y_mid), dist=platform_distance))

        # track reward, max reward is the optimal case
        self.accumulated_reward = 0
        self.last_reward = 0
        self.max_reward = 0
        self.reward_delay = int(floor(platform_distance / com_range)) + 1
        self.time_to_reward = 0

        # observation and action space sizes
        self.total_obs_size = self.get_obs_space().shape[0]
        self.total_actions_size = self.get_action_space().shape[0]
        self.adj_matrix_size = self.n_agents ** 2
        self.agent_obs_size = int((self.total_obs_size - self.adj_matrix_size) / self.n_agents)
        self.agent_action_size = int(self.total_actions_size / self.n_agents)

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
        move = Box(0, 1, shape=(2,)) # relative movement in x and y direction, 0 - 0.33 = left/down, 0.33 - 0.66 = stay, 0.66 - 1 = right/up
        comm_vec = Box(0, 1, shape=(self.size_com_vec,)) # communication vector
        agent_actions = flatten_space(Tuple([move, comm_vec]))

        return flatten_space(Tuple([agent_actions for _ in range(self.n_agents)]))
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """obs space consisting of all agent states + adjacents matrix"""
        platform_location = Box(-self.com_range, self.com_range, shape=(2,)) # relative position of platform
        oracle_location = Box(-self.com_range, self.com_range, shape=(2,)) # relative position of oracle
        platform_occupation = Box(-1, 1, shape=(1,)) # -1 if not visible, else 0/1 if it is occupied
        oracle_state = Box(-1, 1, shape=(1,)) # -1 if not visible, else what the oracle is saying
        comm_vec = Box(0, 1, shape=(self.size_com_vec,)) # communication vector
        agent_state = flatten_space(Tuple([platform_location, oracle_location, platform_occupation, oracle_state, comm_vec]))
        all_agent_states = flatten_space(Tuple([agent_state for _ in range(self.n_agents)]))

        adj_matrix = Box(0, 1, shape=(self.n_agents * self.n_agents,), dtype=np.int8)        
        flat_obs = flatten_space(Tuple([all_agent_states, adj_matrix]))

        return flat_obs
    
    def get_obs(self) -> dict:
        """
        gather information about all agents states and their connectivity.
        fill the observation in the linear obs_space with the same format as described in get_obs_space
        """
        obs = np.zeros(shape=(self.total_obs_size,))
        adj_matrix_offset = self.n_agents * self.agent_obs_size
        for worker in self.schedule.agents:
            obs_offset = worker.unique_id * self.agent_obs_size
            # comm vec
            obs[obs_offset + 6: obs_offset + 6 + self.size_com_vec] = worker.get_comm_vec()
            # positional data 
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.com_range, include_center=True)
            for n in neighbors:
                rel_pos = get_relative_pos(worker.pos, n.pos)
                if type(n) is Platform:
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
        def _decode_action(a: float) -> int:
            if a < 0.33:
                return -1
            elif a < 0.66:
                return 0
            else:
                return 1

        for i, worker in enumerate(self.schedule.agents):
            # move 
            x_old, y_old = worker.pos
            x_new = max(0, min(self.n_tiles_x - 1, x_old + _decode_action(actions[i * self.agent_action_size])))
            y_new = max(0, min(self.n_tiles_y - 1, y_old + _decode_action(actions[i * self.agent_action_size + 1])))
            self.grid.move_agent(worker, (x_new, y_new))
            # comm vector
            worker.set_comm_vec(actions[i * self.agent_action_size + 2: i * self.agent_action_size + 2 + self.size_com_vec])

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
        platform_occupation = self.platform.is_occupied()

        # dont go on platform if oracle is not active
        if not self.oracle.is_active():
            if platform_occupation == 1:
                return -1, 0
            else:
                return 0, 0
        else:
            # time delay to diffuse oracle instruction to all agents
            if self.time_to_reward > 0:
                return 0, 0
            elif oracle_state == 1:
                if platform_occupation == 1:
                    return 1, 1
                else:
                    return -1, 1
            elif oracle_state == 0:
                if platform_occupation == 1:
                    return -1, 0
                else:
                    return 0, 0

    

