import random
import mesa
from math import floor
import numpy as np
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete
from gymnasium.spaces.utils import flatdim

from utils import get_random_pos_on_border, get_relative_pos
from envs.communication_v1.agents import Oracle, Platform, Worker 

MAX_COMMUNICATION_RANGE = 20
TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

class CommunicationV1_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self,
                 max_steps: int,
                 n_workers: int, worker_placement: str,
                 platform_distance: int, oracle_burn_in: int, p_oracle_change: float,
                 n_tiles_x: int, n_tiles_y: int,
                 size_hidden_vec: int, com_range: int, len_trace: int,
                 policy_net: Algorithm = None,
                 platform_placement: str = None) -> None:
        super().__init__()

        self.policy_net = policy_net # not None in inference mode

        self.n_workers = n_workers
        self.size_hidden_vec = size_hidden_vec
        self.com_range = com_range
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        assert com_range <= MAX_COMMUNICATION_RANGE, f"communication range is bigger than MAX_COMMUNICATION_RANGE ({MAX_COMMUNICATION_RANGE}), with which the observation space is computed"
        
        self.max_steps = max_steps
        self.n_steps = 0 # current number of steps
        self.oracle_burn_in = oracle_burn_in
        self.p_oracle_change = p_oracle_change

        # grid coordinates, bottom left = (0,0)
        self.grid = mesa.space.MultiGrid(n_tiles_x, n_tiles_y, False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.current_id = 0

        # map centerpoint
        y_mid = floor(n_tiles_y / 2)
        x_mid = floor(n_tiles_x / 2)
        assert x_mid >= platform_distance and y_mid >= platform_distance, "platform distance to oracle is too large, placement will be out-of-bounds"

        # place oracle in the middle
        self.oracle = Oracle(self._next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_mid, y_mid))
        self.schedule.add(self.oracle)

        # place n platforms around it
        self.n_platforms = 1
        self.platform = Platform(self._next_id(), self)
        platform_distance = platform_distance if platform_placement is None else random.randint(1, platform_distance)
        self.grid.place_agent(agent=self.platform, pos=get_random_pos_on_border(center=(x_mid, y_mid), dist=platform_distance))
        self.schedule.add(self.platform)

        # create workers
        for _ in range(n_workers):
            new_worker = Worker(self._next_id(), self, hidden_vec=np.random.rand(size_hidden_vec))
            self.schedule.add(new_worker)
            self.grid.place_agent(agent=new_worker, pos=(x_mid, y_mid))
            if worker_placement == "random":
                self.grid.move_to_empty(agent=new_worker)

        # track reward, max reward is the optimal case
        self.accumulated_reward = 0
        self.last_reward = 0
        self.max_reward = 0
        self.reward_delay = int(floor(platform_distance / com_range)) + 1
        self.time_to_reward = 0

        # observation and action space sizes
        self.n_total_agents = self.n_workers + self.n_platforms + 1 # workers + platforms + oracle

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
        """
        tuple of action spaces for all workers
        oracle and platform don't get actions or are set manually
        """
        agent_actions = [
            Box(-1, 1, shape=(2,), dtype=np.int32), # move
            Box(0, 1, shape=(self.size_hidden_vec,), dtype=np.float32), # hidden vector
        ]
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """
        obs space consisting of all agent states + adjacents matrix with edge attributes
        keep it as list to keep flexibility in adding/removing attributes
        """
        agent_state = [
            Discrete(3), # agent type
            Box(0, 1, shape=(self.size_hidden_vec,), dtype=np.float32) # hidden vector
        ]
        agent_states = Tuple([Tuple(agent_state) for _ in range(self.n_total_agents)])

        edge_state = [
            Discrete(2), # exists flag
            Box(-MAX_COMMUNICATION_RANGE, MAX_COMMUNICATION_RANGE, shape=(2,), dtype=np.int32), # relative position to the given node
        ]
        edge_states = Tuple([Tuple(edge_state) for _ in range(self.n_total_agents * self.n_total_agents)])

        return Tuple([agent_states, edge_states])
    
    def get_obs(self) -> dict:
        """
        gather information about all agents states and their connectivity.
        fill the observation in the linear obs_space with the same format as described in get_obs_space
        """
        agent_states = [None for _ in range(self.n_total_agents)]
        edge_states = [tuple([0, np.zeros(shape=(2,), dtype=np.int32)]) for _ in range(self.n_total_agents * self.n_total_agents)]
        for i, worker in enumerate(self.schedule.agents):
            # add agent states
            if type(worker) is Oracle:
                oracle_state_one_hot = np.zeros(self.size_hidden_vec, dtype=np.float32)
                oracle_state_one_hot[0] = worker.get_state()
                agent_states[i] = tuple([TYPE_ORACLE, oracle_state_one_hot])
            if type(worker) is Platform:
                platform_occupation = worker.is_occupied()
                agent_states[i] = tuple([TYPE_PLATFORM, np.array([platform_occupation] * self.size_hidden_vec, dtype=np.float32)])
            if type(worker) is Worker:
                agent_states[i] = tuple([TYPE_WORKER, worker.get_hidden_vec()])

            # edge states
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.com_range, include_center=True)
            for n in neighbors:
                rel_pos = get_relative_pos(worker.pos, n.pos)
                edge_states[i * self.n_total_agents + n.unique_id] = tuple([1, np.array(rel_pos, dtype=np.int32)])
        
        assert all([x is not None for x in agent_states]), "agent states are not complete"
        return tuple([tuple(agent_states), tuple(edge_states)])
        
    def apply_actions(self, actions) -> None:
        """apply the actions to the indivdual agents"""

        for i, worker in enumerate([x for x in self.schedule.agents if type(x) is Worker]):
            # decode actions
            x_action, y_action = actions[i][0]
            hidden_vec = actions[i][1]

            # move 
            x_old, y_old = worker.pos
            x_new = max(0, min(self.n_tiles_x - 1, x_old + x_action))
            y_new = max(0, min(self.n_tiles_y - 1, y_old + y_action))
            self.grid.move_agent(worker, (x_new, y_new))
            
            # comm vector
            worker.set_hidden_vec(hidden_vec)

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

    

