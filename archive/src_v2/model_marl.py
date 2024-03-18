from math import floor
from matplotlib import pyplot as plt
import mesa
import random
import numpy as np
import networkx as nx
from ray.rllib.algorithms import Algorithm

import gymnasium
from gymnasium.spaces import Box, Tuple, Discrete
import torch
from agents_marl import BaseAgent, Oracle, Worker
from model_marl_validation import Marl_Lever_Pulling

from utils import compute_agent_placement, get_relative_pos

MAX_DISTANCE = 5000
GRAPH_HASH = 1000000

TYPE_ORACLE = 0
TYPE_PLATFORM = 1
TYPE_WORKER = 2

def get_model_by_config(config: str):
    if config in [
                    "env_config_10.yaml",
                    "env_config_11.yaml",
                    "env_config_12.yaml",
                ]: return Marl_model
    elif config in [
                    "env_config_13.yaml",
                ]: return Relstate_Model
    elif config in [
                    "env_config_14.yaml",
                    "env_config_15.yaml",
                    "env_config_16.yaml",
                ]: return Moving_Discrete_model
    elif config in [
                    "env_config_17.yaml",
                    "env_config_18.yaml",
                    "env_config_19.yaml",
                    "env_config_20.yaml",
                    "env_config_21.yaml",
                    "env_config_22.yaml",
                    "env_config_23.yaml",
                    "env_config_24.yaml",
                    "env_config_25.yaml",
                    "env_config_26.yaml",
                    "env_config_graph.yaml",
                ]: return Moving_History_model
    elif config in [
        "env_config_lever.yaml",
    ]: return Marl_Lever_Pulling

class Marl_model(mesa.Model):
    """
    the oracle outputs a number which the agents need to copy
    the visibility range is limited, so the agents need to communicate the oracle output
    reward: how many agents have the correct output
    """

    def __init__(self, config: dict,
                 use_cuda: bool = False,
                 policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__()

        # workers
        self.n_workers = config["model"]["n_workers"]
        self.n_hidden_states = config["model"]["n_hidden_state"]
        self.communication_range = config["model"]["communication_range"]
        self.worker_placement = config["model"]["worker_placement"]
        self.worker_init = config["model"]["worker_init"]
        self.reward_calculation = config["model"]["reward_calculation"]
        self.n_agents = self.n_workers + 1
        # oracle
        self.n_oracle_states = config["model"]["n_oracle_states"]
        self.p_oracle_change = config["model"]["p_oracle_change"]
        # grid
        self.grid_size = config["model"]["grid_size"]
        self.grid_middle = floor(self.grid_size / 2)
        # misc
        self.episode_length = config["model"]["episode_length"]
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # tracking
        self.ts_episode = 0
        self.ts_curr_state = 0
        self.state_switch_pause = floor(self.grid_middle / self.communication_range) + 1
        self.n_state_switches = 1
        self.reward_total = 0
        self.reward_lower_bound = 0
        self.reward_upper_bound = 0
        
        # mesa setup
        self.grid = mesa.space.MultiGrid(self.grid_size, self.grid_size, False)
        self.schedule_all = mesa.time.BaseScheduler(self)
        self.schedule_workers = mesa.time.BaseScheduler(self)
        self.current_id = 0

        # initialisation outputs of agents
        oracle_output = random.randint(0, self.n_oracle_states-1)
        if self.worker_init == "uniform":
            r = random.randint(0, self.n_oracle_states-1)
            while r == oracle_output:
                r = random.randint(0, self.n_oracle_states-1)
            worker_output = [r for _ in range(self.n_workers)]
        else:
            worker_output = [random.randint(0, self.n_oracle_states-1) for _ in range(self.n_workers)]

        # place agents
        self.oracle = Oracle(unique_id=self._next_id(), 
                             model=self, 
                             output=oracle_output,
                             n_hidden_states=self.n_hidden_states)
        oracle_pos = (self.grid_middle, self.grid_middle)
        self.grid.place_agent(agent=self.oracle, pos=oracle_pos)
        self.schedule_all.add(self.oracle)
        agent_positions = compute_agent_placement(self.n_workers, self.communication_range, 
                                                  self.grid_size, 
                                                  oracle_pos, self.worker_placement)
        for i, curr_pos in enumerate(agent_positions):
            worker = Worker(unique_id=self._next_id(), 
                            model=self, 
                            output=worker_output[i], 
                            n_hidden_states=self.n_hidden_states)
            self.grid.place_agent(agent=worker, pos=curr_pos)
            self.schedule_all.add(worker)
            self.schedule_workers.add(worker)

        # inference mode
        self.inference_mode = inference_mode
        self.policy_net = policy_net

    def _next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def _compute_reward(self):
        assert self.reward_calculation in {"individual", "per-agent"}

        # compute reward
        rewardss = {}
        n_wrongs = sum([1 for a in self.schedule_workers.agents if a.output != self.oracle.output])
        if self.reward_calculation == "individual":
            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = -n_wrongs if n_wrongs else self.n_workers
            upper = self.n_workers * self.n_workers
            lower = -self.n_workers * self.n_workers
        elif self.reward_calculation == "per-agent":
            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = 1 if worker.output == self.oracle.output else -1
            upper = self.n_workers
            lower = -self.n_workers

        return rewardss, upper, lower, n_wrongs
    
    def get_action_space(self) -> gymnasium.spaces.Space:
        """action space per agent"""
        return Tuple([
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])
    
    def _apply_action(self, agent: BaseAgent, action):
        agent.output = action[0]
        agent.hidden_state = action[1]
    
    def get_obs_space(self) -> gymnasium.spaces.Space:
        """ obs space consisting of all agent states + adjacents matrix with edge attributes """
        graph_state = Box(0, GRAPH_HASH, shape=(1,), dtype=np.float32) # current step hash
        agent_states = Tuple([self._get_agent_state_space() for _ in range(self.n_agents)])
        edge_states = Tuple([self._get_edge_state_space() for _ in range(self.n_agents * self.n_agents)])
        return Tuple([graph_state, agent_states, edge_states])
    
    def _get_edge_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2), # exists flag
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32), # relative position to the given node
        ])
    
    def _get_edge_state(self, from_agent: BaseAgent, to_agent: BaseAgent, visible_edge: int):
        return tuple([visible_edge, np.array(get_relative_pos(from_agent.pos, to_agent.pos))])
    
    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                # active flag
            Discrete(3),                                                # agent type
            Discrete(self.n_oracle_states),                             # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
        ])

    def _get_agent_state(self, agent: BaseAgent, activity_status: int):
        """compute agent state"""
        return tuple([activity_status,
                        TYPE_ORACLE if type(agent) is Oracle else TYPE_WORKER, 
                        agent.output,
                        agent.hidden_state])
        
    def get_obss(self) -> dict:
        """ collect and return current obs"""
        step_hash = np.array([random.randint(0, GRAPH_HASH)])

        # agent states
        agent_states = [None for _ in range(self.n_agents)]
        for worker in self.schedule_all.agents:
            agent_states[worker.unique_id] = self._get_agent_state(agent=worker, activity_status=0)
            
        # edge attributes
        edge_states = [None for _ in range(self.n_agents ** 2)]
        for worker in self.schedule_all.agents:
            # fully connected graph
            for destination in self.schedule_all.agents:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visible_edge=0)

            # visible graph
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for destination in neighbors:
                edge_states[worker.unique_id * self.n_agents + destination.unique_id] = self._get_edge_state(from_agent=worker, to_agent=destination, visible_edge=1)


        # make obs_space for every agent, changing the live flag
        obss = dict()
        for worker in self.schedule_workers.agents:
            curr_agent_state = agent_states.copy()
            curr_agent_state[worker.unique_id] = self._get_agent_state(agent=worker, activity_status=1)
            obss[worker.unique_id] = tuple([step_hash, tuple(curr_agent_state), tuple(edge_states)])
        return obss

    def get_graph(self, save_fig=False):
        """compute adjacency graph"""
        graph = nx.Graph()
        positioning = {}
        for worker in self.schedule_all.agents:
            graph.add_node(worker.unique_id)
            positioning[worker.unique_id] = np.array(worker.pos)
        for worker in self.schedule_all.agents:
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            for n in neighbors:
                graph.add_edge(worker.unique_id, n.unique_id)

        if save_fig:
            graph.remove_edges_from(list(nx.selfloop_edges(graph)))
            nx.draw_networkx_edges(graph, positioning, 
                                    arrows=True,
                                   connectionstyle='arc3, rad = 0.5')
            nx.draw_networkx_nodes(graph, positioning, 
                                    node_size=350,
                                    node_color=['green'] + ['black'] * (self.n_workers))
            nx.draw_networkx_labels(graph, positioning, 
                                    font_size=16, font_color="white")
            plt.show()
        return graph
    
    def _print_model_specific(self):
        pass
        

    def step(self, actions=None) -> None:
        """advance the model one step in inference mode"""        
        # determine actions for inference mode
        if self.inference_mode:
            actions = dict()
            obss = self.get_obss()

            if self.policy_net:
                for worker in self.schedule_workers.agents:
                    #actions[worker.unique_id] = self.policy_net.compute_single_action(obss[worker.unique_id])
                    actions[worker.unique_id], _, _ = self.policy_net.compute_single_action(obss[worker.unique_id], state=np.array([worker.unique_id, self.n_workers]))
            else:
                for worker in self.schedule_workers.agents:
                    actions[worker.unique_id] = self.get_action_space().sample()

        # advance simulation
        for k, v in actions.items():
            worker = [w for w in self.schedule_all.agents if w.unique_id == k][0]
            self._apply_action(agent=worker, action=v)

        self.ts_episode += 1
        self.ts_curr_state += 1
        self.running = self.ts_episode < self.episode_length
        
        # compute reward and state
        rewardss, dupper, dlower, n_wrongs = self._compute_reward()
        self.reward_total += sum(rewardss.values())
        self.reward_upper_bound += dupper
        self.reward_lower_bound += dlower

        truncateds = {"__all__": self.ts_episode >= self.episode_length}
        terminateds = {"__all__": False}

        # terminate or change to new oracle state
        old_output = self.oracle.output
        ts_old_output = self.ts_curr_state
        if self.p_oracle_change > 0 and self.running and \
            ts_old_output >= self.state_switch_pause and self.ts_episode + self.state_switch_pause <= self.episode_length:
            r = random.random()
            if r <= self.p_oracle_change:
                new_state = random.randint(0, self.n_oracle_states-1)
                while new_state == self.oracle.output:
                    new_state = random.randint(0, self.n_oracle_states-1)
                self.oracle.output = new_state
                self.n_state_switches += 1
                self.ts_curr_state = 0

        # print overview
        if self.inference_mode:
            print()
            print(f"------------- step {self.ts_episode}/{self.episode_length} ------------")
            print(f"  outputs            = {old_output} - {[a.output for a in self.schedule_workers.agents]}")
            print(f"  rewards            = {rewardss}")
            print(f"  converged          = {n_wrongs == 0}")
            print(f"  truncated          = {self.ts_episode >= self.episode_length}")
            print()
            print(f"  next_state         = {'-' if old_output == self.oracle.output else {self.oracle.output}}")
            print(f"  state_switch_pause = {ts_old_output}/{self.state_switch_pause}")
            print(f"  n_state_switches   = {self.n_state_switches}")
            print()
            print(f"  reward_lower_bound = {self.reward_lower_bound}")
            print(f"  reward_total       = {self.reward_total}")
            print(f"  reward_upper_bound = {self.reward_upper_bound}")
            print(f"  reward_percentile  = {(self.reward_total - self.reward_lower_bound) / (self.reward_upper_bound - self.reward_lower_bound)}")
            print()
            self._print_model_specific()

        return self.get_obss(), rewardss, terminateds, truncateds


class Relstate_Model(Marl_model):
    
    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(2),                                                    # active flag
            Discrete(3),                                                    # agent type
            Discrete(self.n_oracle_states),                                 # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32),     # hidden state
            Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32), # relative position to oracle
        ])
    
    def _get_agent_state(self, agent: BaseAgent, activity_status: int):
        """compute agent state"""
        return tuple([
            activity_status,
            TYPE_ORACLE if type(agent) is Oracle else TYPE_WORKER, 
            agent.output,
            agent.hidden_state,
            np.array(get_relative_pos(agent.pos, self.oracle.pos))
        ])
    
    def _print_model_specific(self):
        obss = self.get_obss()
        print("  worker relative positions:")
        for agent in self.schedule_all.agents:
            print(f"    {agent.name} {agent.pos}: ", obss[1][1][agent.unique_id][4])
        #self.get_graph(save_fig=True)

        print("  edges:")
        for worker in self.schedule_all.agents:
            neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
            neighbors = [n for n in neighbors if n != worker]
            for destination in neighbors:
                print(f"    edge {worker.unique_id}->{destination.unique_id}: {self._get_edge_state(from_agent=worker, to_agent=destination, visible_edge=1)}")
        print()


class Moving_Discrete_model(Marl_model):
    def get_action_space(self) -> gymnasium.spaces.Space:
        return Tuple([
            Discrete(self.n_oracle_states),                             # output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32), # hidden state
            Box(-1, 1, shape=(2,), dtype=np.int32),                     # movement x,y
        ]) 
    
    def _apply_action(self, agent: BaseAgent, action):
        agent.output = action[0]
        agent.hidden_state = action[1]
        dx, dy = action[2]
        dx = -1 if dx <= -0.3 else 1 if dx >= 0.3 else 0
        dy = -1 if dy <= -0.3 else 1 if dy >= 0.3 else 0
        x = max(0, min(self.grid_size-1, agent.pos[0] + dx))
        y = max(0, min(self.grid_size-1, agent.pos[1] + dy))
        self.grid.move_agent(agent=agent, pos=(x,y))

    def _compute_reward(self):
        assert self.reward_calculation in {"spread", "spread-connected", "2-neighbours", "scn2", "scn3", "graph-validation"}

        # compute reward
        rewardss = {}
        n_wrongs = sum([1 for a in self.schedule_workers.agents if a.output != self.oracle.output])
        if self.reward_calculation == "spread":
            for worker in self.schedule_workers.agents:
                dx, dy = get_relative_pos(worker.pos, self.oracle.pos)
                rewardss[worker.unique_id] = max(abs(dx), abs(dy)) if worker.output == self.oracle.output else -1
            
            lower = -self.n_workers
            upper = 0
            for i in range(self.n_workers):
                upper += min((i+1) * self.communication_range, self.grid_middle)
            
        elif self.reward_calculation == "spread-connected":
            g = self.get_graph()
            for worker in self.schedule_workers.agents:
                dx, dy = get_relative_pos(worker.pos, self.oracle.pos)
                if worker.output == self.oracle.output:
                    rewardss[worker.unique_id] = max(abs(dx), abs(dy)) * (1 if nx.has_path(g, self.oracle.unique_id, worker.unique_id) else 0.5)
                else:
                    rewardss[worker.unique_id] = -1 * (0.5 if nx.has_path(g, self.oracle.unique_id, worker.unique_id) else 1)

            lower = -self.n_workers
            upper = 0
            for i in range(self.n_workers):
                upper += min((i+1) * self.communication_range, self.grid_middle)

        elif self.reward_calculation == "2-neighbours":
            for worker in self.schedule_workers.agents:
                neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
                neighbors = [n for n in neighbors if n != worker]

                if worker.output == self.oracle.output:
                    if 0 < len(neighbors) < 3:
                        reward = 1
                    elif len(neighbors) >= 3:
                        reward = 0.5
                    else:
                        reward = 0.1
                else:
                    if 0 < len(neighbors) < 3:
                        reward = -0.2
                    elif len(neighbors) >= 3:
                        reward = -0.5
                    else:
                        reward = -1
                rewardss[worker.unique_id] = reward
            
            lower = -self.n_workers
            upper = self.n_workers
        elif self.reward_calculation == "scn2":
            for worker in self.schedule_workers.agents:
                g = self.get_graph()
                dx, dy = get_relative_pos(worker.pos, self.oracle.pos)

                # find neighbours factor
                neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
                neighbors = [n for n in neighbors if n != worker]
                is_connected = nx.has_path(g, self.oracle.unique_id, worker.unique_id)
    
                # reward
                if worker.output == self.oracle.output:
                    if 0 < len(neighbors) < 3:
                        reward = max(abs(dx), abs(dy)) if is_connected else max(abs(dx), abs(dy)) / 10
                    elif len(neighbors) >= 3:
                        reward = max(abs(dx), abs(dy)) / 5 * 4 if is_connected else max(abs(dx), abs(dy)) / 10
                    else:
                        reward = -0.1
                else:
                    reward = -1
                rewardss[worker.unique_id] = reward
            
            lower = -self.n_workers
            upper = 0
            for i in range(self.n_workers):
                upper += min((i+1) * self.communication_range, self.grid_middle)
        elif self.reward_calculation == "scn3":
            for worker in self.schedule_workers.agents:
                g = self.get_graph()
                dx, dy = get_relative_pos(worker.pos, self.oracle.pos)

                # find neighbours factor
                neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
                neighbors = [n for n in neighbors if n != worker]
                is_connected = nx.has_path(g, self.oracle.unique_id, worker.unique_id)
    
                # reward
                if worker.output == self.oracle.output:
                    if 0 < len(neighbors) < 3:
                        reward = max(abs(dx), abs(dy)) if is_connected else max(abs(dx), abs(dy)) / 10
                    elif len(neighbors) >= 3:
                        reward = max(abs(dx), abs(dy)) / 5 * 4 if is_connected else max(abs(dx), abs(dy)) / 10
                    else:
                        reward = -0.1
                else:
                    reward = -1
                rewardss[worker.unique_id] = reward
            
            lower = -self.n_workers
            upper = self.n_workers * self.grid_middle / 2

        elif self.reward_calculation == "graph-validation":
            reward = 0
            lower = 0
            upper = 0
            m_wrongs = 0
            for worker in self.schedule_workers.agents:
                # find neighbours factor
                neighbors = self.grid.get_neighbors(worker.pos, moore=True, radius=self.communication_range, include_center=True)
                neighbors = [n for n in neighbors if n != worker]
    
                for n in neighbors:
                    dx, dy = get_relative_pos(worker.pos, n.pos)
                    dist = max(abs(dx), abs(dy))
                    if dist < 3:
                        reward += -2.5
                    else:
                        reward += 0.5

                    upper += 0.5 * self.n_workers
                    lower += -2.5 * self.n_workers

            for worker in self.schedule_workers.agents:
                rewardss[worker.unique_id] = reward
            
        return rewardss, upper, lower, n_wrongs

class Moving_History_model(Moving_Discrete_model):

    def __init__(self, config: dict, use_cuda: bool = False, policy_net: Algorithm = None, inference_mode: bool = False) -> None:
        super().__init__(config, use_cuda, policy_net, inference_mode)
        self.history_length = 3
        self.history = {w.unique_id: [[np.array(get_relative_pos(w.pos, self.oracle.pos)), np.array([0,0])] for _ in range(self.history_length)] 
                        for w in self.schedule_all.agents}
    
    def _get_agent_state_space(self) -> gymnasium.spaces.Space:
        history = [Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32)]
        for _ in range(self.history_length):
            history.append(Box(-MAX_DISTANCE, MAX_DISTANCE, shape=(2,), dtype=np.float32))
            history.append(Box(-1, 1, shape=(2,), dtype=np.int32))

        return Tuple([
            Discrete(2),                                                    # active flag
            Discrete(3),                                                    # agent type
            Discrete(self.n_oracle_states),                                 # current output
            Box(0, 1, shape=(self.n_hidden_states,), dtype=np.float32)]     # hidden state                               
            + history)
    
    def _apply_action(self, agent: BaseAgent, action):
        # update history
        h = self.history[agent.unique_id]
        h.insert(0, [np.array(get_relative_pos(agent.pos, self.oracle.pos)), action[2]])
        if len(h) > self.history_length: h.pop()
        
        # apply action
        super()._apply_action(agent=agent, action=action)

    
    def _get_agent_state(self, agent: BaseAgent, activity_status: int):
        """compute agent state"""
        history = [np.array(get_relative_pos(agent.pos, self.oracle.pos))]
        for ht in self.history[agent.unique_id]:
            pos, a = ht
            history.append(pos)
            history.append(a)
        return tuple([
            activity_status,
            TYPE_ORACLE if type(agent) is Oracle else TYPE_WORKER, 
            agent.output,
            agent.hidden_state]
            + history)

    def _print_model_specific(self):
            print("positional history")
            print("------------------")
            hist = [[] for _ in range(4)]
            for agent in self.schedule_workers.agents:
                for j, h in enumerate(self.history[agent.unique_id]):
                    hist[j].append(str(h[0]) + " " + str(h[1]))
            
            print("\t\t\t".join(["agent " + str(w.unique_id) for w in self.schedule_workers.agents]))
            print("\t\t".join(["pos|action " for _ in self.schedule_workers.agents]))
            for ht in hist:
                print("\t\t".join(ht))
            
            print()