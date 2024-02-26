import random
import torch
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from math import floor
import mesa
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete
from ray.rllib.algorithms import Algorithm

from agents import BaseAgent, Oracle, Worker
from task_utils import get_worker_outputs, get_worker_placements, get_relative_pos

MAX_DISTANCE = 5000
GRAPH_HASH = 1000000

class base_model(mesa.Model):
    def __init__(self, config: dict,
                 use_cuda: bool = False,
                 inference_mode: bool = False,
                 policy_net: Algorithm = None) -> None:
        super().__init__()

        # base init
        self.config = config
        self.inference_mode = inference_mode
        self.policy_net = policy_net

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.current_id = 0

    def _next_id(self) -> int:
        """return the next unique agent ID"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def _print_state(self):
        """print model state in inference mode"""
        raise NotImplementedError("_print_state not yet implemented")
        
    def _compute_reward(self):
        """compute reward from the current task state"""
        raise NotImplementedError("_compute_reward not yet implemented")
    
    def _apply_action(self, agent: BaseAgent, action):
        """apply action of the agent to the model"""
        raise NotImplementedError("_apply_action not yet implemented")
    
    def _get_edge_state_space(self) -> gymnasium.spaces.Space:
        """
        return obs space of one edge
        important: the first subspace of the tuple must be Discrete(2), marking the visibility of an edge
        """
        raise NotImplementedError("_get_edge_state_space not yet implemented")
    
    def _get_edge_state(self, from_agent: BaseAgent, to_agent: BaseAgent, visibility: int):
        """return state of the edge (from_agent, to_agent), mark it's visibility"""
        raise NotImplementedError("_get_edge_state not yet implemented")

    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        """
        return obs space of one agent
        important: the first subspace of the tuple must be Discrete(2), marking the if agent is active
        """
        raise NotImplementedError("_get_agent_state_space not yet implemented")
    
    def _get_agent_state(self, agent: BaseAgent, active: int):
        """return obs of agent, mark it's activity"""
        raise NotImplementedError("_get_agent_state not yet implemented")

    def get_action_space(self) -> gymnasium.spaces.Space:
        """return action space of model"""
        raise NotImplementedError("get_action_space not yet implemented")

    def get_obs_space(self) -> gymnasium.spaces.Space:
        """return obs space of model"""
        raise NotImplementedError("get_obs_space not yet implemented")
    
    def get_obs(self):
        """return observation of the current model state"""
        raise NotImplementedError("get_obs not yet implemented")
    
    def as_graph(self, save_fig: str = None):
        """return the current model state as nx.Graph()"""
        raise NotImplementedError("as_graph not yet implemented")

    def step(self, action=None):
        """
        advance model:
            actions are None     -> inference mode, sample actions randomly or from policy_net
            actions are not None -> apply
        """
        raise NotImplementedError("step not yet implemented")


class lever_pulling_model(base_model):
    def __init__(self, config: dict,
                 use_cuda: bool = False,
                 inference_mode: bool = False,
                 policy_net: Algorithm = None) -> None:
        super().__init__(config=config, use_cuda=use_cuda, inference_mode=inference_mode, policy_net=policy_net)

        self.n_agents = 5
        self.agent_id_max = 500
        self.n_levers = 5

        # to not break the pipeline
        self.reward_total = 0
        self.reward_lower_bound = 0
        self.reward_upper_bound = 0
        self.n_state_switches = 0
        self.grid = mesa.space.MultiGrid(1, 1, False)

    def get_action_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(self.n_levers),
        ])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        graph_state = Box(0, int(GRAPH_HASH), shape=(1,), dtype=np.float32)     
        agent_states = Tuple([Tuple([
            Discrete(2),                                                       
            Discrete(self.agent_id_max)     # only deciding factor                                      
        ]) for _ in range(self.n_agents)])
        edge_states = Tuple([Tuple([
            Discrete(2),                                                       
            Box(0, 1, shape=(2,), dtype=np.float32),
        ]) for _ in range(self.n_agents * self.n_agents)])
        
        return Tuple([graph_state, agent_states, edge_states])
    
    def get_obs(self):
        step_hash = np.array([random.randint(0, GRAPH_HASH)])
        agent_ids = [random.randint(0, self.agent_id_max - 1) for _ in range(self.n_agents)]
        agent_states = [tuple([0, agent_ids[i]]) for i in range(self.n_agents)]
        edge_states = [tuple([1, np.array([0,0])]) for _ in range(self.n_agents ** 2)]
        
        obss = dict()
        for i in range(self.n_agents):
            curr_agent_state = agent_states.copy()
            curr_agent_state[i] = tuple([1, agent_ids[i]])
            obss[i] = tuple([step_hash, tuple(curr_agent_state), tuple(edge_states)])
        return obss
    
    def step(self, actions=None):

        # determine actions for inference mode
        if self.inference_mode:
            actions = dict()
            obss = self.get_obs()
            if self.policy_net:
                for i in range(self.n_agents):
                    actions[i], _, _ = self.policy_net.compute_single_action(obss[i], state=np.array([i, self.n_agents]))
            else:
                for i in range(self.n_agents):
                    actions[i] = self.get_action_space().sample()

        levers = actions.values()
        r = float(len(set(levers))) / self.n_levers
        self.reward_total = r
        self.reward_upper_bound = 1

        if self.inference_mode:
            print("pulled levers: ")
            print(levers)
            print("reward: ", r)
        return self.get_obs(), {a: r for a in range(self.n_agents)}, {"__all__": True}, {"__all__": False}

class oracle_model(base_model):
    """
    tasks which have one oracle and n worker agents in a grid world
        oracle is central and outputs a number
        workers need to copy this number to get reward
        visibility range of workers is limited
    """
    def __init__(self, config: dict,
                 use_cuda: bool = False,
                 inference_mode: bool = False,
                 policy_net: Algorithm = None) -> None:
        super().__init__(config=config, use_cuda=use_cuda, inference_mode=inference_mode, policy_net=policy_net)

        # worker configs
        self.n_workers = config["model"]["n_workers"]
        self.n_hidden_states = config["model"]["n_hidden_state"]
        self.communication_range = config["model"]["communication_range"]
        self.worker_placement = config["model"]["worker_placement"]
        self.worker_init = config["model"]["worker_init"]
        self.reward_calculation = config["model"]["reward_calculation"]
        self.n_agents = self.n_workers + 1

        # oracle configs
        self.n_oracle_states = config["model"]["n_oracle_states"]
        self.p_oracle_change = config["model"]["p_oracle_change"]
        
        # grid config
        self.grid_size = config["model"]["grid_size"]
        self.grid_middle = floor(self.grid_size / 2)
        
        # task config
        self.episode_length = config["model"]["episode_length"]

        # tracking
        self.reward_last = 0
        self.reward_total = 0
        self.reward_lower_bound = 0
        self.reward_upper_bound = 0
        self.ts_episode = 0
        self.ts_curr_state = 0
        self.state_switch_pause = floor(self.grid_middle / self.communication_range) + 1
        self.n_state_switches = 1
        
        # mesa setup
        self.grid = MultiGrid(self.grid_size, self.grid_size, False)
        self.schedule_all = BaseScheduler(self)
        self.schedule_workers = BaseScheduler(self)

        # oracle setup
        oracle_output = random.randint(0, self.n_oracle_states-1)
        self.oracle = Oracle(unique_id=self._next_id(), 
                             model=self, 
                             output=oracle_output,
                             n_hidden_states=self.n_hidden_states)
        oracle_pos = (self.grid_middle, self.grid_middle)
        self.grid.place_agent(agent=self.oracle, pos=oracle_pos)
        self.schedule_all.add(self.oracle)

        # agent setup
        worker_outputs = get_worker_outputs(self.n_workers,
                                            self.n_oracle_states,
                                            oracle_output,
                                            self.worker_init)
        worker_positions = get_worker_placements(self.n_workers, 
                                                self.communication_range, 
                                                self.grid_size, 
                                                oracle_pos, 
                                                self.worker_placement)
        for i, curr_pos in enumerate(worker_positions):
            worker = Worker(unique_id=self._next_id(), 
                            model=self, 
                            output=worker_outputs[i], 
                            n_hidden_states=self.n_hidden_states)
            self.grid.place_agent(agent=worker, pos=curr_pos)
            self.schedule_all.add(worker)
            self.schedule_workers.add(worker)

    def _get_worker_neighbours(self, worker: BaseAgent, include_self: bool):
        """compute all agents in the neighbourhood of worker"""
        neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
        if not include_self:
            neighbors = [n for n in neighbors if n != worker]
        return neighbors

    def as_graph(self, save_fig: str = None):
        graph = nx.Graph()
        for worker in self.schedule_all.agents:
            graph.add_node(worker.unique_id)
        for worker in self.schedule_all.agents:
            neighbors = self._get_worker_neighbours(worker, include_self=False)
            for n in neighbors:
                graph.add_edge(worker.unique_id, n.unique_id)

        if save_fig:
            positioning = {}
            for worker in self.schedule_all.agents:
                positioning[worker.unique_id] = np.array(worker.pos)

            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            nx.draw_networkx_edges(graph, 
                                   positioning, 
                                   arrows=True,
                                   connectionstyle='arc3, rad = 0.5')
            nx.draw_networkx_nodes(graph, 
                                   positioning,  
                                   node_size=350, 
                                   node_color=['green'] + ['black'] * (self.n_workers))
            nx.draw_networkx_labels(graph, 
                                    positioning, 
                                    font_size=16, 
                                    font_color="white")
            # @todo: save graph output to path save_fig
            plt.show()

        return graph
    
class transmission_model_marl(oracle_model):
    def _compute_reward(self):
        """ note: bounds must be multiplied by number of agents as the reward is the sum of all agent rewards """
        assert self.reward_calculation in {"shared_binary", "shared_sum", "individual"}, f"reward calculation {self.reward_calculation} not implemented for {type(self)}"

        rewardss = {}
        wrongs = sum([1 for worker in self.schedule_workers.agents if worker.output != self.oracle.output])

        # shared sparse feedback if swarm state is correct or not
        if self.reward_calculation == "shared_binary":
            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = -1 if wrongs else 1
            upper = self.n_workers
            lower = -self.n_workers
        # shared feedback how well the swarm is doing
        elif self.reward_calculation == "shared_sum":
            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = -wrongs if wrongs else self.n_workers
            upper = self.n_workers * self.n_workers
            lower = -self.n_workers * self.n_workers
        # each agent get's individual feedback
        elif self.reward_calculation == "individual":
            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = 1 if worker.output == self.oracle.output else -1
            upper = self.n_workers
            lower = -self.n_workers

        return rewardss, upper, lower
    
    def _apply_action(self, agent: BaseAgent, action):
        agent.output = action[0]
        agent.hidden_state = action[1]
    
    def _get_edge_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                    # visible edge flag
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32), # relative distance between nodes
        ])
    
    def _get_edge_state(self, from_agent: BaseAgent, to_agent: BaseAgent, visibility: int):
        return tuple([
            visibility, 
            np.array(get_relative_pos(from_agent.pos, to_agent.pos))
        ])
    
    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                # active agent flag
            Discrete(3),                                                # agent type
            Discrete(self.n_oracle_states),                             # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])

    def _get_agent_state(self, agent: BaseAgent, active: int):
        return tuple([
            active,
            agent.type, 
            agent.output,
            agent.hidden_state
            ])
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """
        action space per agent view:
            no movement
        """
        return Tuple([
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """
        obs space per agent view:
            graph hash to cache computed graphs in gnn-module 
            all agent states + edge states to be able to independently build graph
        
            Note: this observation is still per agent view, exactly one agent will have it's activity flag set
                the activity flag is later used in the gnn-module to determine who queried the output, e.g. which node wants it's result (because each agent only wants it's own action in MARL)
        """
        graph_state = Box(0, GRAPH_HASH, shape=(1,), dtype=np.float32)
        agent_states = Tuple([self._get_agent_state_space() for _ in range(self.n_agents)])
        edge_states = Tuple([self._get_edge_state_space() for _ in range(self.n_agents * self.n_agents)])
        return Tuple([graph_state, agent_states, edge_states])
    
        
    def get_obs(self) -> dict:
        """
        compute observation dict, including observation for each agent:
            key:    agent who queries
            value:  the whole state, with the querying agent set as active
        """
        graph_hash = np.array([random.randint(0, GRAPH_HASH)])
        
        agent_states = [None for _ in range(self.n_agents)]
        for worker in self.schedule_all.agents:
            agent_states[worker.unique_id] = self._get_agent_state(agent=worker, active=0)

        edge_states = [None for _ in range(self.n_agents ** 2)]
        for worker in self.schedule_all.agents:
            # build full graph 
            for destination in self.schedule_all.agents:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visibility=0)
            # activate visible edges only
            neighbors =  self._get_worker_neighbours(worker=worker, include_self=False)
            for destination in neighbors:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visibility=1)

        # create agent specific obs spaces
        obss = dict()
        for worker in self.schedule_workers.agents:
            curr_agent_state = agent_states.copy()
            curr_agent_state[worker.unique_id] = self._get_agent_state(agent=worker, active=1)
            obss[worker.unique_id] = tuple([graph_hash, tuple(curr_agent_state), tuple(edge_states)])
        return obss
    
    def _print_model_specific(self):
        pass
        

    def step(self, actions=None) -> None:
        # sample actions for inference mode
        if self.inference_mode:
            actions = dict()
            obss = self.get_obs()

            for worker in self.schedule_workers.agents:
                if self.policy_net:
                    # abuse state parameter to enable cashing of the computation graph in gnn module
                    #   unfortunately no batch evaluation is possible, e.g. the cashing mechanism in learning mode is not available
                    actions[worker.unique_id], _, _ = self.policy_net.compute_single_action(obss[worker.unique_id], state=np.array([worker.unique_id, self.n_workers]))
                else:
                    actions[worker.unique_id] = self.get_action_space().sample()

        # apply agent actions to model
        for k, v in actions.items():
            worker = [w for w in self.schedule_all.agents if w.unique_id == k][0]
            self._apply_action(agent=worker, action=v)

        # compute reward and update tracking
        rewardss, upper, lower = self._compute_reward()
        self.reward_last = sum(rewardss.values())
        self.reward_total += sum(rewardss.values())
        self.reward_upper_bound += upper
        self.reward_lower_bound += lower
        self.ts_episode += 1
        self.ts_curr_state += 1
        self.running = self.ts_episode < self.episode_length

        truncateds = {"__all__": self.ts_episode >= self.episode_length}
        terminateds = {"__all__": False}

        # change oracle state
        oracle_output_old = self.oracle.output
        ts_old = self.ts_curr_state
        if self.p_oracle_change > 0 and self.running and \
            ts_old >= self.state_switch_pause and self.ts_episode + self.state_switch_pause <= self.episode_length:
            r = random.random()
            if r <= self.p_oracle_change:
                new_state = random.randint(0, self.n_oracle_states-1)
                while new_state == self.oracle.output:
                    new_state = random.randint(0, self.n_oracle_states-1)
                self.oracle.output = new_state
                self.n_state_switches += 1
                self.ts_curr_state = 0

        # print status
        if self.inference_mode:
            print()
            print(f"------------- step {self.ts_episode}/{self.episode_length} ------------")
            print(f"  outputs            = {oracle_output_old} - {[a.output for a in self.schedule_workers.agents]}")
            print(f"  rewards            = {self.reward_last} - {rewardss}")
            print()
            print(f"  reward_total       = {self.reward_total}")
            print(f"  reward_lower_bound = {self.reward_lower_bound}")
            print(f"  reward_upper_bound = {self.reward_upper_bound}")
            print(f"  reward_percentile  = {(self.reward_total - self.reward_lower_bound) / (self.reward_upper_bound - self.reward_lower_bound)}")
            print()
            print(f"  next_state         = {'-' if oracle_output_old == self.oracle.output else {self.oracle.output}}")
            print(f"  state_switch_pause = {ts_old}/{self.state_switch_pause}")
            print(f"  n_state_switches   = {self.n_state_switches}")
            print()
            self._print_model_specific()

        return self.get_obs(), rewardss, terminateds, truncateds
    
class transmission_model_rl(oracle_model):
    def _compute_reward(self):
        assert self.reward_calculation in {"shared_binary", "shared_sum"}, f"reward calculation {self.reward_calculation} not implemented for {type(self)}"

        wrongs = sum([1 for worker in self.schedule_workers.agents if worker.output != self.oracle.output])

        # shared sparse feedback if swarm state is correct or not
        if self.reward_calculation == "shared_binary":
            reward = -1 if wrongs else 1
            upper = 1
            lower = -1
        # shared feedback how well the swarm is doing
        elif self.reward_calculation == "shared_sum":
            reward = -wrongs if wrongs else self.n_workers
            upper = self.n_workers
            lower = -self.n_workers

        return reward, upper, lower
    
    def _apply_action(self, agent: BaseAgent, action):
        agent.output = action[0]
        agent.hidden_state = action[1]
    
    def _get_edge_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                    # visible edge flag
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32), # relative distance between nodes
        ])
    
    def _get_edge_state(self, from_agent: BaseAgent, to_agent: BaseAgent, visibility: int):
        return tuple([
            visibility, 
            np.array(get_relative_pos(from_agent.pos, to_agent.pos))
        ])
    
    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                # active agent flag (unused in gnn module, left if for compatibility purposes)
            Discrete(3),                                                # agent type
            Discrete(self.n_oracle_states),                             # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])

    def _get_agent_state(self, agent: BaseAgent, active: int):
        return tuple([
            active,
            agent.type, 
            agent.output,
            agent.hidden_state
            ])
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """action space including all agents actions"""
        agent_actions = Tuple([
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])
        return Tuple([Tuple(agent_actions) for _ in range(self.n_workers)])
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """
        obs space per graph:
            graph hash not used in gnn-module, but left for compatibility 
            all agent states + edge states to build graph
        """
        graph_state = Box(0, GRAPH_HASH, shape=(1,), dtype=np.float32)
        agent_states = Tuple([self._get_agent_state_space() for _ in range(self.n_agents)])
        edge_states = Tuple([self._get_edge_state_space() for _ in range(self.n_agents * self.n_agents)])
        return Tuple([graph_state, agent_states, edge_states])
    
        
    def get_obs(self) -> dict:
        """compute observation of the whole graph"""
        graph_hash = np.array([random.randint(0, GRAPH_HASH)])  # unused in gnn module, left in for compatibility purposes
        
        agent_states = [None for _ in range(self.n_agents)]
        for worker in self.schedule_all.agents:
            agent_states[worker.unique_id] = self._get_agent_state(agent=worker, active=1)

        edge_states = [None for _ in range(self.n_agents ** 2)]
        for worker in self.schedule_all.agents:
            # build full graph 
            for destination in self.schedule_all.agents:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visibility=0)
            # activate visible edges only
            neighbors =  self._get_worker_neighbours(worker=worker, include_self=False)
            for destination in neighbors:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visibility=1)

        return tuple([graph_hash, tuple(agent_states), tuple(edge_states)])
    
    def _print_model_specific(self):
        pass
        

    def step(self, actions=None) -> None:
        # sample actions for inference mode
        if self.inference_mode:
            if self.policy_net:
                actions = self.policy_net.compute_single_action(self.get_obs())
            else:
                actions = self.get_action_space().sample()

        # apply agent actions to model
        for i, worker in enumerate(self.schedule_workers.agents):
            self._apply_action(agent=worker, action=actions[i])

        # compute reward and update tracking
        reward, upper, lower = self._compute_reward()
        self.reward_last = reward
        self.reward_total += reward
        self.reward_upper_bound += upper
        self.reward_lower_bound += lower
        self.ts_episode += 1
        self.ts_curr_state += 1
        self.running = self.ts_episode < self.episode_length

        # change oracle state
        oracle_output_old = self.oracle.output
        ts_old = self.ts_curr_state
        if self.p_oracle_change > 0 and self.running and \
            ts_old >= self.state_switch_pause and self.ts_episode + self.state_switch_pause <= self.episode_length:
            r = random.random()
            if r <= self.p_oracle_change:
                new_state = random.randint(0, self.n_oracle_states-1)
                while new_state == self.oracle.output:
                    new_state = random.randint(0, self.n_oracle_states-1)
                self.oracle.output = new_state
                self.n_state_switches += 1
                self.ts_curr_state = 0

        # print status
        if self.inference_mode:
            print()
            print(f"------------- step {self.ts_episode}/{self.episode_length} ------------")
            print(f"  outputs            = {oracle_output_old} - {[a.output for a in self.schedule_workers.agents]}")
            print(f"  rewards            = {self.reward_last}")
            print()
            print(f"  reward_total       = {self.reward_total}")
            print(f"  reward_lower_bound = {self.reward_lower_bound}")
            print(f"  reward_upper_bound = {self.reward_upper_bound}")
            print(f"  reward_percentile  = {(self.reward_total - self.reward_lower_bound) / (self.reward_upper_bound - self.reward_lower_bound)}")
            print()
            print(f"  next_state         = {'-' if oracle_output_old == self.oracle.output else {self.oracle.output}}")
            print(f"  state_switch_pause = {ts_old}/{self.state_switch_pause}")
            print(f"  n_state_switches   = {self.n_state_switches}")
            print()
            self._print_model_specific()

        return self.get_obs(), reward, False, self.ts_episode >= self.episode_length