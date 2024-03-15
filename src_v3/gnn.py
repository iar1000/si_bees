from typing import Dict, List
import numpy as np
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.profile import count_parameters
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch_geometric.nn import Sequential as PyG_Sequential, global_mean_pool
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

from environment import MARL_ENV
from task_utils import get_graph_from_batch_obs

class gnn_torch_module(TorchModelV2, Module):
    def __init__(self, 
                    obs_space: Space,
                    action_space: Space,
                    num_outputs: int,
                    model_config: dict,
                    name: str,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        Module.__init__(self)

        # configs
        config = model_config["custom_model_config"]
        self.env_type = config["env_type"]
        actor_config = config["actor_config"]
        critic_config = config["critic_config"]
        encoding_config = config["encoding_config"]
        self.encoding_size = encoding_config["encoding_size"]
        self.recurrent_actor = config["recurrent_actor"]
        self.recurrent_critic = config["recurrent_critic"]

        # graph lookup for marl training
        self.eval_duration = None
        self.eval_lookup = None

        # model dimensions
        og_obs_space = obs_space.original_space
        self.num_inputs = flatdim(og_obs_space)
        self.num_agents = len(og_obs_space[1])
        self.adj_matrix_size = len(og_obs_space[2])
        self.node_state_size = flatdim(og_obs_space[1][0]) - 2 # activity status of agent is not part of the state
        self.edge_state_size = flatdim(og_obs_space[2][0]) - 2 # visibility of an edge is not part of the state
        # in RL, the action space is the individual action space of all workers concatenated 
        self.out_state_size = num_outputs if self.env_type == MARL_ENV else num_outputs // (self.num_agents - 1)

        self.node_encoder = self.__build_fc(ins=self.node_state_size, outs=self.encoding_size, hiddens=[], activation=None)
        self.edge_encoder = self.__build_fc(ins=self.edge_state_size, outs=self.encoding_size, hiddens=[], activation=None)
        self.action_decoder = self.__build_fc(ins=self.encoding_size + self.node_state_size if self.recurrent_actor else self.encoding_size, 
                                       outs=self.out_state_size, 
                                       hiddens=[], 
                                       activation="tanh")
        self.value_decoder = self.__build_fc(ins=self.encoding_size + self.node_state_size if self.recurrent_critic else self.encoding_size, 
                                       outs=1, 
                                       hiddens=[], 
                                       activation=None)
        self.actor = self._build_module(config=actor_config, 
                                       ins=self.encoding_size, 
                                       outs=self.encoding_size, 
                                       edge_dim=self.encoding_size, 
                                       add_pooling=False)
        self.critic = self._build_module(config=critic_config, 
                                        ins=self.encoding_size, 
                                        outs=self.encoding_size, 
                                        edge_dim=self.encoding_size, 
                                        add_pooling=False)
        
        # put to correct device for gpu training
        config["use_cuda"] = True
        self.device = torch.device("cuda:0" if config["use_cuda"] else "cpu")
        self.node_encoder.to(device=self.device)
        self.edge_encoder.to(device=self.device)
        self.actor.to(device=self.device)
        self.critic.to(device=self.device)

        print(f"env type: ", self.env_type)
        print(f"actor ({next(self.actor.parameters()).device}): ", self.actor)
        print(f"critic ({next(self.critic.parameters()).device}): ", self.critic)
        print(f"node encoder ({next(self.node_encoder.parameters()).device}): ", self.node_encoder)
        print(f"edge encoder ({next(self.edge_encoder.parameters()).device}): ", self.edge_encoder)
        print(f"node state size: ", self.node_state_size)
        print(f"edge state size: ", self.edge_state_size)
        print(f"encoding size: ", self.encoding_size)
        print(f"action size: ", self.out_state_size)
        print(f"recurrent_actor: ", self.recurrent_actor)
        print(f"recurrent_critic: ", self.recurrent_critic)
        print(f"device: ", self.device)
        print(f"  cuda_is_available={torch.cuda.is_available()}")
        print(f"  use_cuda={config['use_cuda']}")
        print(f"num parameters: ", count_parameters(self.node_encoder), count_parameters(self.actor), count_parameters(self.critic))

        
    def __build_fc(self, ins: int, outs: int, hiddens: list, activation: str):
        """builds a fully connected network """
        layers = list()
        prev_layer_size = ins
        for curr_layer_size in hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=activation))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))
        return Sequential(*layers)      

    def __build_gnn(self, config: dict, ins: int, outs: int, edge_dim: int):
        """builds one gnn layer"""
        if config["model"] == "GATConv":
            return GATConv(ins, outs, dropout=config["dropout"])
        elif config["model"] == "GATv2Conv":
            return GATv2Conv(ins, outs, edge_dim=edge_dim, dropout=config["dropout"])
        elif config["model"] == "GINEConv":
            return GINEConv(self.__build_fc(ins, outs, [config["hidden_mlp_size"] for _ in range(config["n_hidden_mlp"])], activation="relu"))
        elif config["model"] == "TransformerConv":
            return TransformerConv(ins, outs, edge_dim=edge_dim, dropout=config["dropout"])
        else:
            raise NotImplementedError(f"unknown model {config['model']}")  
        
    def _build_module(self, config: dict, ins: int, outs: int, edge_dim: int, add_pooling: bool) -> Module:
        """builds a module consisting of multiple gnn rounds"""
        gnn_rounds = list()
        for _ in range(config["rounds"] - 1):
            gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
            gnn_rounds.append(torch.nn.ReLU(inplace=True))
        
        # last layer, pooling if necessecary
        if add_pooling:
            gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
            gnn_rounds.append((global_mean_pool, 'x, batch -> x'))
        else:
            gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=outs, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   

        return PyG_Sequential('x, edge_index, edge_attr, batch', gnn_rounds)
  
    def value_function(self):
        return torch.reshape(self.last_values, [-1])
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        """
        extract the node and edge info
        encode node and edge info
        build adjacency relations
        compute outputs with _actor and _critic modules

        assumes that one sample is of the form (graph_hash, agent_obss, edge_obss)
            in RL: graph hash and agent active status are ignored
        """
        obss = input_dict["obs"]
        obss_flat = input_dict["obs_flat"]
        graph_hashs = obss[0]
        agent_obss = obss[1]
        edge_obss = obss[2]
        batch_size = len(obss_flat)

        # MARL evaluation mode graph caching
        #   cache graph result to not recompute the same graph for each agent
        #   use state argument to derive when to cache and when to kick 
        eval_mode = state and len(state) == 2
        if eval_mode:
            if self.eval_lookup:
                action = self.eval_lookup[state[0].item()]
                self.eval_duration -= 1
                if self.eval_duration == 0:
                    self.eval_lookup = None
                return action, []
            else:
                self.eval_lookup = dict()
                self.eval_duration = state[1] - 1

        # MARL training mode graph caching
        curr_graph_index = 0        # index for seen graphs                                    
        hash_to_graph_index = {}    # map hash to internal index
        hash_to_num_queries = {}    # for debugging, story how many agents queried the same graph        
        # go through samples and find active agent of each sample, to later map the right node-state to the right sample
        sample_to_node_index = {}
        for sample_id in range(batch_size):
            for agent_id in range(len(agent_obss)):
                if agent_obss[agent_id][0][sample_id][1] == 1:
                    sample_to_node_index[sample_id] = agent_id

        # build graphs from the batch
        actor_graphs_old = list()
        actor_graphs = list()
        fc_graphs = list()
        for i in range(batch_size):
            
            # only build graph once if multiple agents query their action for the same graph
            if graph_hashs[i].item() in hash_to_graph_index.keys():
                hash_to_num_queries[graph_hashs[i].item()] += 1
                continue
            else:
                hash_to_graph_index[graph_hashs[i].item()] = curr_graph_index
                hash_to_num_queries[graph_hashs[i].item()] = 1
                curr_graph_index += 1

            # get graph data for sample i from batched observations
            x, actor_edge_index, actor_edge_attr, fc_edge_index, fc_edge_attr = get_graph_from_batch_obs(self.num_agents, agent_obss, edge_obss, i) 
            # encode node and edge states
            x = [e.to(self.device) for e in x]
            x_old = torch.clone(torch.stack([v for v in x]))
            x_old = [e.to(self.device) for e in x_old]
            x = torch.stack([self.node_encoder(v) for v in x])
            
            actor_edge_index = torch.tensor(actor_edge_index, dtype=torch.int64, device=self.device)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.int64, device=self.device)
            if actor_edge_attr:
                actor_edge_attr = [e.to(self.device) for e in actor_edge_attr]
                actor_edge_attr = torch.stack([self.edge_encoder(e) for e in actor_edge_attr]) 
            else:
                actor_edge_attr = torch.zeros((0, self.encoding_size), dtype=torch.float32, device=self.device)

            if fc_edge_attr:
                fc_edge_attr = [e.to(self.device) for e in fc_edge_attr]
                fc_edge_attr = torch.stack([self.edge_encoder(e) for e in fc_edge_attr])
            else:
                fc_edge_attr = torch.zeros((0, self.encoding_size), dtype=torch.float32, device=self.device)

            actor_graphs_old.append(Data(x=x_old, edge_index=actor_edge_index, edge_attr=actor_edge_attr))
            actor_graphs.append(Data(x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr))
            fc_graphs.append(Data(x=x, edge_index=fc_edge_index, edge_attr=fc_edge_attr))
            
        # create superbatch
        actor_old_dataloader = DataLoader(dataset=actor_graphs_old, batch_size=batch_size)
        actor_dataloader = DataLoader(dataset=actor_graphs, batch_size=batch_size)
        critic_dataloader = DataLoader(dataset=fc_graphs, batch_size=batch_size)
        actor_old_batch = next(iter(actor_old_dataloader))
        actor_batch = next(iter(actor_dataloader))
        critic_batch = next(iter(critic_dataloader))
        assert torch.all(actor_batch.batch.eq(actor_old_batch.batch)), "something went wrong, the recurrent copy of the graph should be the same"
        assert torch.all(actor_batch.batch.eq(critic_batch.batch)), "something went wrong, the recurrent copy of the graph should be the same"

        # compute actor hidden states and decode actions
        actor_h_all = self.actor(x=actor_batch.x, edge_index=actor_batch.edge_index, edge_attr=actor_batch.edge_attr, batch=actor_batch.batch)
        if self.recurrent_actor:
            actions_all = self.action_decoder(torch.cat([actor_h_all, actor_old_batch.x], dim=1))
        else:
            actions_all = self.action_decoder(actor_h_all)

        # compute critic hidden states and decode values
        critic_h_all = self.critic(x=critic_batch.x, edge_index=critic_batch.edge_index, edge_attr=critic_batch.edge_attr, batch=critic_batch.batch)
        if self.recurrent_critic:
            critic_h_all = torch.cat([critic_h_all, actor_old_batch.x], dim=1)
        
        # MARL: compute value for each node
        # RL  : take the mean of all critic_h to compute one value per graph, but this later, so no need to compute value for each node
        values_all = self.value_decoder(critic_h_all)
              
        # reverse superbatching: compute start and end index of results for each sub-graph
        #   i.e. actions_all[from:to] came from the same sub-graph
        curr_graph_index = 0
        result_to_graph_index_list = [0]
        for i, subgraph_index in enumerate(actor_batch.batch):
            # results are belonging to the next subgraph, i.e. start of next subgraph at this index
            if curr_graph_index != subgraph_index:
                curr_graph_index += 1
                result_to_graph_index_list.append(i)
        result_to_graph_index_list.append(len(actions_all))

        # split up actions_all in actions per graph with index list
        actions_per_graph = list()
        values_per_graph = list()
        for start_index in range(len(result_to_graph_index_list) - 1):
            curr_graph_actions = actions_all[result_to_graph_index_list[start_index]:result_to_graph_index_list[start_index+1]]
            curr_graph_values = values_all[result_to_graph_index_list[start_index]:result_to_graph_index_list[start_index+1]]
            curr_graph_critic_h = critic_h_all[result_to_graph_index_list[start_index]:result_to_graph_index_list[start_index+1]]
            # MARL: each graph has n_workers independend actions and values
            if self.env_type == MARL_ENV:
                actions_per_graph.append(curr_graph_actions)
                values_per_graph.append(curr_graph_values)
            # RL: each graph has concatenated actions output of all agents and one value
            else:
                actions_per_graph.append(torch.flatten(curr_graph_actions[1:]))
                values_per_graph.append(self.value_decoder(torch.mean(curr_graph_critic_h, dim=0)))

        # print(f"batch_size: {batch_size}")
        # print(f"actor_h_all output: {actor_h_all.shape}")
        # print(f"critic_h_all output: {critic_h_all.shape}")
        # print(f"actions_all output: {actions_all.shape}")
        # print(f"values_all output: {values_all.shape}")
        # print(f"hash to graph: {hash_to_graph_index}")
        # print(f"sample to sample_to_node_index: {sample_to_node_index}")
        # print(f"num graphs: {len(hash_to_graph_index)}: {sum([1 for k in hash_to_num_queries.keys() if hash_to_num_queries[k] == 4])}/{sum([1 for k in hash_to_num_queries.keys() if hash_to_num_queries[k] == 3])}/{sum([1 for k in hash_to_num_queries.keys() if hash_to_num_queries[k] == 2])}/{sum([1 for k in hash_to_num_queries.keys() if hash_to_num_queries[k] == 1])}")
        # print("action per graph: ", len(actions_per_graph))
        # print("action per graph: ", len(actions_per_graph[0]))
        # print("action per graph: ", len(values_per_graph))
        # print("action per graph: ", len(values_per_graph[0]))

        # return dummy values in initialisation run
        # note: carefull, this means that at least one agent must be active to not fall into this. it is only supposed to handle the dummy call in the beginning
        if not sample_to_node_index:
            # MARL: actions_per_graph holds multiple actions
            if self.env_type == MARL_ENV:
                actions = torch.stack([actions_per_graph[0][0] for _ in range(batch_size)])
            # RL: actions_per_graph holds one time the concatentaion of all actions
            else:
                actions = torch.stack([actions_per_graph[0] for _ in range(batch_size)])
                
            self.last_values = torch.stack([torch.tensor(np.zeros(1), device=self.device) for _ in range(batch_size)])
            return actions, []
        else:
            # MARL: per graph there is an action output per node, which need to returned to the correct sample
            if self.env_type == MARL_ENV:
                actions = list()
                values = list()
                for i in range(batch_size):
                    graph_index = hash_to_graph_index[graph_hashs[i].item()]
                    actions_of_batch = actions_per_graph[graph_index]
                    values_of_batch = values_per_graph[graph_index]
                    # return the action of the active agent for this query
                    actions.append(actions_of_batch[sample_to_node_index[i]])
                    values.append(values_of_batch[sample_to_node_index[i]])
                
                # cache all actions outputed by graph
                if eval_mode:
                    for i, s in enumerate(actions_of_batch):
                        self.eval_lookup[i] = torch.stack([s])
            # RL: each graph has only one action output and value output
            else:
                actions = actions_per_graph
                values = values_per_graph

            self.last_values = torch.stack(values)
            return torch.stack(actions), []