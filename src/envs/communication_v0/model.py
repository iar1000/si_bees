import gymnasium
import mesa
from math import floor

from envs.communication_v0.agents import Plattform, Worker, Oracle

class CommunicationV0_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self, config, policy_net=None) -> None:
        super().__init__()
        self.config = config
        self.policy_net = policy_net # not None in inference mode

        # key coordinates
        x_max = self.config["mesa_grid_width"] - 1
        y_max = self.config["mesa_grid_height"] - 1
        y_mid = floor(y_max / 2)
        x_mid = floor(x_max / 2)

        # create model
        # grid coordinates:
        #   bottom left = (0,0)
        #   top right = (height-1, width-1)
        self.grid = mesa.space.MultiGrid(self.config["mesa_grid_width"], self.config["mesa_grid_height"], False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.possible_agents = []
        self.agent_name_to_id = {}

        # track number of rounds, steps tracked by scheduler
        self.n_rounds = 0
        self.total_reward = 0
        self.last_reward = 0

        # create workers
        for _ in range(self.config["num_agents"]):
            new_worker = Worker(self.next_id(), self, 
                                n_hidden_vec=self.config["n_hidden_vec"],
                                n_comm_vec=self.config["n_comm_vec"],
                                n_visibility_range=self.config["n_visibility_range"],
                                n_comm_range=self.config["n_comm_range"],
                                n_trace_length=self.config["n_trace_length"])
            self.schedule.add(new_worker)
            self.grid.place_agent(agent=new_worker, pos=(x_mid,0))
            
            self.possible_agents.append(new_worker.name)
            self.agent_name_to_id[new_worker.name] = new_worker.unique_id

        # create oracle and lightswitch
        margin = 2
        x_oracle = margin
        self.oracle = Oracle(self.next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_oracle, y_mid))

        x_plattform = x_max - margin
        self.plattform = Plattform(self.next_id(), self)
        self.grid.place_agent(agent=self.plattform, pos=(x_plattform, y_mid))

        # track time for the information to travel
        self.comm_distance = x_plattform - x_oracle
        self.reward_delay = self.comm_distance # @todo: if the agents can see further than 1 square, this needs to be smaller
        self.time_to_reward = 0

    def next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def print_status(self) -> None:
        """print status of the model"""
        print(f"round {self.n_rounds}: oracle is {'off' if not self.oracle.is_active() else 'on'}\n\ttime to reward={self.time_to_reward}\n\treward={self.last_reward}, acc_reward={self.total_reward}")

    
    def finish_round(self) -> int:
        """
        finish up a round
        - increases the round counter by 1
        - change oracle state
        - count points
        """
        # update round
        self.n_rounds += 1
        self.time_to_reward = max(0, self.time_to_reward - 1)

        self.last_reward = self.compute_reward()
        self.total_reward += self.last_reward

        # activate oracle
        if not self.oracle.is_active() and self.config["oracle_burn_in"] < self.n_rounds:
            r = self.random.random()
            if r > self.config["p_oracle_activation"]:
                self.oracle.set_state(1)
                self.oracle.activate()
                self.time_to_reward = self.reward_delay

        # @todo: change state of oracle
        
        return self.n_rounds
    
    def compute_reward(self) -> int:
        """computes the reward based on the current state"""
        oracle_state = self.oracle.get_state()
        plattform_occupation = len(self.plattform.get_occupants()) > 0
        
        # dont go on plattform if oracle is not active
        if not self.oracle.is_active():
            if plattform_occupation == 1:
                return -1
            else:
                return 0
        else:
            # time delay to diffuse oracle instruction to all agents
            if self.time_to_reward > 0:
                return 0
            # actual reward of actions
            elif oracle_state == 0 and plattform_occupation == 1 or \
                oracle_state == 1 and plattform_occupation == 0:
                return -1
            elif oracle_state == 1 and oracle_state == plattform_occupation:
                return 10
            else:
                return 1

    def get_num_rounds_and_steps(self) -> [int, int]:
        """returns number of rounds and step of the currend model"""
        return self.n_rounds, self.schedule.steps

    def get_oracle_and_plattform(self) -> [Oracle, Plattform]:
        """returns oracle and plattform agents"""
        return self.oracle, self.plattform

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

    def step_agent(self, agent_id, action) -> None:
         """applies action to the agent in the environment"""
         agent = self.schedule.agents[agent_id]
         agent.step(action=action)

    def step(self) -> None:
        """step once through all agents, used for inference"""
        self.schedule.step()

        next_round = self.finish_round()
        self.print_status()
        if next_round >= self.config["mesa_max_rounds"]:
            self.running = False

    def observe_agent(self, agent_id) -> dict:
        """returns the observation of the agent in the current model state"""
        agent = self.schedule.agents[agent_id]
        return agent.observe()
    

