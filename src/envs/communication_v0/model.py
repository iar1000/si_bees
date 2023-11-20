import gymnasium
import mesa
from math import floor

from ray.rllib.algorithms import Algorithm 

from envs.communication_v0.agents import Platform, Worker, Oracle
from utils import get_random_pos_on_border

class CommunicationV0_model(mesa.Model):
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
                 policy_net: Algorithm = None) -> None:
        super().__init__()

        self.policy_net = policy_net # not None in inference mode

        self.max_steps = max_steps
        self.oracle_burn_in = oracle_burn_in
        self.p_oracle_change = p_oracle_change

        assert n_agents > 0, "must have a positive number of agents"
        assert agent_placement in ["center", "random"], f"agent placement {agent_placement} unknown"
        assert n_tiles_x > 2 and n_tiles_y > 2, "minimum map size of 3x3 required"
        assert 0 < platform_distance, "distance of platform to oracle must be at least 1"

        # create model
        # grid coordinates, bottom left = (0,0)
        self.grid = mesa.space.MultiGrid(n_tiles_x, n_tiles_y, False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.possible_agents = []
        self.agent_name_to_id = {}

        # map centerpoint
        y_mid = floor(n_tiles_y / 2)
        x_mid = floor(n_tiles_x / 2)
        assert x_mid >= platform_distance and y_mid >= platform_distance, "platform distance to oracle is too large, placement will be out-of-bounds"

        # create workers
        for _ in range(n_agents):
            new_worker = Worker(self._next_id(), self, 
                                size_com_vec=size_com_vec,
                                com_range=com_range,
                                len_trace=len_trace)
            self.schedule.add(new_worker)
            self.possible_agents.append(new_worker.name)
            self.agent_name_to_id[new_worker.name] = new_worker.unique_id

            # place workers
            self.grid.place_agent(agent=new_worker, pos=(x_mid, y_mid))
            if agent_placement == "random":
                self.grid.move_to_empty(agent=new_worker)

        # place oracle in the middle and lightswitch around it
        self.oracle = Oracle(self._next_id(), self)
        self.platform = Platform(self._next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_mid, y_mid))
        self.grid.place_agent(agent=self.platform, pos=get_random_pos_on_border(center=(x_mid, y_mid), dist=platform_distance))

        # track number of rounds, steps tracked by scheduler
        self.n_steps = 0

        # track reward, max reward is the optimal case
        self.accumulated_reward = 0
        self.last_reward = 0
        self.max_reward = 0
        self.reward_delay = int(floor(platform_distance / com_range)) + 1
        self.time_to_reward = 0


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
        for agent_name in self.possible_agents:
            agent_id = self.agent_name_to_id[agent_name]
            agent = self.schedule.agents[agent_id]
            out += f"{agent_name}: {agent.pos} "
        print(out)

    def get_possible_agents(self) -> [list, dict]:
        """returns list of scheduled agent names and dict to map names to respective ids"""
        return self.possible_agents, self.agent_name_to_id
    
    def get_action_space(self, agent_id) -> gymnasium.spaces.Space:
        agent = self.schedule.agents[agent_id]
        return agent.get_action_space()
    
    def get_obs_space(self, agent_id) -> gymnasium.spaces.Space:
        agent = self.schedule.agents[agent_id]
        return agent.get_obs_space()
    
    def has_policy(self) -> bool:
        """returns if an action can be sampled from a policy net"""
        return self.policy_net is not None

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


    def step_agent(self, agent_id, action) -> None:
         """applies action to the agent in the environment"""
         agent = self.schedule.agents[agent_id]
         agent.step(action=action)

    def observe_agent(self, agent_id) -> dict:
        """returns the observation of the agent in the current model state"""
        agent = self.schedule.agents[agent_id]
        return agent.observe()

    # for running the model in inference mode over the webserver
    def step(self) -> None:
        """step once through all agents, used for inference"""
        self.schedule.step()
        self.finish_round()
        self.print_status()

    

