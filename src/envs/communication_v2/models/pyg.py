from typing import Dict, List
import numpy as np
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.nn import Sequential as PyG_Sequential, global_mean_pool
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space, Box, Tuple, Discrete
from gymnasium.spaces.utils import flatdim

from envs.communication_v2.model import MAX_DISTANCE
from utils import build_graph_v2
SUPPORTED_ENCODERS = ["identity", "fc", "sincos"]
SUPPORTED_ACTORS = ["GATConv", "GATv2Conv", "GINEConv"]
SUPPORTED_CRITICS = ["fc", "GATConv", "GATv2Conv", "GINEConv"]

GNN_EDGE_WEIGHT_SUPPORT = ["SimpleConv", "GCNConv", "ChebConv", "GraphConv", "GatedGraphConv", "TAGConv", "ARMAConv", "SGConv", "SSGConv", "APPNP", "DNAConv", "LEConv", "GCN2Conv", "WLConvContinuous", "FAConv", "LGConv", "MixHopConv"]
GNN_EDGE_ATTRIBUTE_SUPPORT = ["ResGatedGraphConv", "GATConv", "GATv2Conv", "TransformerConv", "GINEConv", "GMMConv", "SplineConv", "NNConv", "CGConv", "PNAConv", "GENConv", "PDNConv", "GeneralConv"]

class GNN_PyG(TorchModelV2, Module):
    """
    base class for one-round gnn models.
    implements following process:
    """
    def __init__(self, 
                    obs_space: Space,
                    action_space: Space,
                    num_outputs: int,
                    model_config: dict,
                    name: str,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        Module.__init__(self)

        # custom parameters are passed via the model_config dict from ray
        self.custom_config = self.model_config["custom_model_config"]
        self.actor_config = self.custom_config["actor_config"]
        self.critic_config = self.custom_config["critic_config"]
        self.encoders_config = self.custom_config["encoders_config"]

        # check if actor, critic and encoder configs are speciyin supported models
        assert self.actor_config["model"] in SUPPORTED_ACTORS, f"actor model {self.actor_config['model']} not supported"
        assert self.critic_config["model"] in SUPPORTED_CRITICS, f"critic model {self.critic_config['model']} not supported"
        assert self.encoders_config["node_encoder"] in SUPPORTED_ENCODERS, f"node encoder model {self.encoders_config['node_encoder']} not supported"
        assert self.encoders_config["edge_encoder"] in SUPPORTED_ENCODERS, f"edge encoder model {self.encoders_config['edge_encoder']} not supported"

        # model dimensions
        og_obs_space = obs_space.original_space
        self.num_inputs = flatdim(og_obs_space)
        self.num_agents = len(og_obs_space[0])
        self.adj_matrix_size = len(og_obs_space[1])
        self.node_state_size = flatdim(og_obs_space[0][0])
        self.edge_state_size = flatdim(og_obs_space[1][0])
        
        self.n_non_worker_agents = 1
        self.num_outputs = num_outputs
        self.num_outputs_per_agent = num_outputs // (self.num_agents - self.n_non_worker_agents)
        self.num_outputs_per_agent_per_subspace = list()
        for subspace in action_space[0]:
            if type(subspace) == Discrete:
                self.num_outputs_per_agent_per_subspace.append(flatdim(subspace))
            elif type(subspace) == Box:
                self.num_outputs_per_agent_per_subspace.append(flatdim(subspace) * 2)
            else:
                raise Exception(f"don't know how this space translates to the backend model: {type(subspace)}")
        assert sum(self.num_outputs_per_agent_per_subspace) == self.num_outputs_per_agent, f"translation of the action_space to number output parameters is wrong: {self.num_outputs_per_agent}/{self.num_outputs_per_agent_split}"
        
        # encoders
        self._node_encoder, self._edge_encoder, self.encoding_size = self._build_encoders(
            config=self.encoders_config, 
            node_state_size=self.node_state_size,
            edge_state_size=self.edge_state_size)
        # actors
        self._actors = list()
        for num_outs in self.num_outputs_per_agent_per_subspace:
            self._actors.append(self._build_model(
                config=self.actor_config, 
                ins=self.encoding_size, 
                outs=num_outs,
                edge_dim=self.encoding_size,
                add_pooling=False
            ))
        # critic
        self._critic_model_fc = self.critic_config["model"] == "fc"
        self._critic_graph_fc = self.critic_config["critic_fc"]
        self._critic = self._build_model(
            config=self.critic_config, 
            ins=self.num_inputs if self._critic_model_fc else self.encoding_size,
            outs=1,
            edge_dim=self.encoding_size,
            add_pooling=True)
        self.last_values = None
        
        print("\n=== backend model ===")
        print(f"num_agents        = {self.num_agents}")
        print(f"num_inputs        = {self.num_inputs}")
        print(f"num_outputs       = {self.num_outputs}")
        print(f"num_outputs_per_agent = {self.num_outputs_per_agent}")
        print(f"num_outputs_per_agent_per_subspace = {self.num_outputs_per_agent_per_subspace}")
        print(f"adj_matrix_size   = {self.adj_matrix_size}")
        print(f"node_state_size   = {self.node_state_size}")
        print(f"edge_state_size   = {self.edge_state_size}")
        print(f"encoding size     = {self.encoding_size}")
        print("encoders: ", self.encoders_config)
        print("actors:")
        for i, a in enumerate(self._actors):
            print(f" actor {i}: {a}")
        print("critic: ", self._critic)
        print()

    def __build_fc(self, ins: int, outs: int, hiddens: list):
        """builds a fully connected network with relu activation"""
        layers = list()
        prev_layer_size = ins
        for curr_layer_size in hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))
        return Sequential(*layers)

    def __build_gnn(self, config: dict, ins: int, outs: int, edge_dim: int):
        """creates one instance of the in the config specified gnn"""
        if config["model"] == "GATConv":
            return GATConv(ins, outs, dropout=config["dropout"])
        elif config["model"] == "GATv2Conv":
            return GATv2Conv(ins, outs, edge_dim=edge_dim, dropout=config["dropout"])
        elif config["model"] == "GINEConv":
            return GINEConv(self.__build_fc(ins, outs, [config["mlp_hiddens_size"] for _ in range(config["mlp_hiddens"])]))
        else:
            raise NotImplementedError(f"unknown model {config['model']}")
        
    def _build_encoders(self, config: dict, node_state_size: int, edge_state_size: int) -> (Module, Module, int):
        """builds and returns the encoder networks for node or edge values and the encoding size"""
        node_encoder = None
        edge_encoder = None
        encoding_size = config["encoding_size"]

        # node encoder
        if config["node_encoder"] == "fc":
            node_encoder = self.__build_fc(
                ins=node_state_size, 
                outs=encoding_size, 
                hiddens=[config["node_encoder_hiddens_size"] for _ in range(config["node_encoder_hiddens"])])
        else:
            raise NotImplementedError(f"unknown node encoder {config['node_encoder']}")
        
        # edge encoder
        if config["edge_encoder"] == "fc":
            edge_encoder = self.__build_fc(
                ins=edge_state_size, 
                outs=encoding_size, 
                hiddens=[config["edge_encoder_hiddens_size"] for _ in range(config["edge_encoder_hiddens"])])
        elif config["edge_encoder"] == "sincos":
            # lookup
            self.pos_encoding_size = encoding_size // 2 # concat x and y positional encodings, e.g. each pos_encoding only half as big as encoding_size
            max_sequnce_length = 2 * MAX_DISTANCE + 1
            angle_lookup = [1 / (10000 ** (2 * i / self.pos_encoding_size)) for i in range(self.pos_encoding_size // 2)]
            self.sin_encodings = [[np.sin(seq_num * angle_lookup[i]) for seq_num in range(max_sequnce_length)] for i in range(self.pos_encoding_size // 2)]
            self.cos_encodings = [[np.cos(seq_num * angle_lookup[i]) for seq_num in range(max_sequnce_length)] for i in range(self.pos_encoding_size // 2)]
            self.rel_to_sequence = lambda relative_position: relative_position + MAX_DISTANCE

            def sincos(edge_attr):
                encoding = torch.empty(size=(self.encoding_size,))
                for i in range(self.pos_encoding_size // 2):
                    encoding[2*i] = self.sin_encodings[i][self.rel_to_sequence(int(edge_attr[2]))]
                    encoding[2*i + 1] = self.cos_encodings[i][self.rel_to_sequence(int(edge_attr[2]))]
                    encoding[2*i + (self.encoding_size // 2)] = self.sin_encodings[i][self.rel_to_sequence(int(edge_attr[3]))]
                    encoding[2*i + 1 + (self.encoding_size // 2)] = self.cos_encodings[i][self.rel_to_sequence(int(edge_attr[3]))]
                return encoding
            edge_encoder = sincos
        else:
            raise NotImplementedError(f"unknown edge encoder {config['edge_encoder']}")
        
        return node_encoder, edge_encoder, encoding_size

    def _build_model(self, config: dict, ins: int, outs: int, edge_dim: int, add_pooling: bool) -> Module:
        """creates an NN model based on the config dict"""
        if config["model"] != "fc":
            gnn_rounds = list()
            for _ in range(config["rounds"] - 1):
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
                gnn_rounds.append(torch.nn.ReLU(inplace=True))
            
            # add last layer, pooling if necessecary
            if add_pooling:
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
                gnn_rounds.append((global_mean_pool, 'x, batch -> x'))
                gnn_rounds.append(SlimFC(in_size=ins, out_size=outs))
            else:
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=outs, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   

            return PyG_Sequential('x, edge_index, edge_attr, batch', gnn_rounds)
        elif config["model"] == "fc":
            return self.__build_fc(ins=ins, outs=outs, hiddens=[config["mlp_hiddens_size"] for _ in range(config["mlp_hiddens"])])
        else:
            raise NotImplementedError(f"unknown model {config['model']}")        
    
    def value_function(self):
        return torch.reshape(self.last_values, [-1])
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        extract the node info from the flat observation to create an X tensor
        extract the adjacency relations to create the edge_indexes
        feed it to the _actor and _critic methods to get the outputs, those methods are implemented by a subclass

        note: the construction of the graph is tightly coupled to the format of the obs_space defined in the model class
        """    
        outs = []
        values = []

        obss = input_dict["obs"]
        obss_flat = input_dict["obs_flat"]
        agent_obss = obss[0]
        edge_obss = obss[1]
        batch_size = len(obss_flat)

        # iterate through the batch
        for i in range(batch_size):
            x, actor_edge_index, actor_edge_attr, fc_edge_index, fc_edge_attr = build_graph_v2(self.num_agents, agent_obss, edge_obss, i) 
            
            # format graph to torch
            x = torch.stack([self._node_encoder(v) for v in x])
            actor_edge_index = torch.tensor(actor_edge_index, dtype=torch.int64)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.int64)
            actor_edge_attr = torch.stack([self._edge_encoder(e) for e in actor_edge_attr]) if actor_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)
            fc_edge_attr = torch.stack([self._edge_encoder(e) for e in fc_edge_attr]) if fc_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)

            # compute results of all individual actors and concatenate the results
            all_actions = self._actors[0](x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr, batch=torch.zeros(x.shape[0],dtype=int))
            all_actions = [actor(x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr, batch=torch.zeros(x.shape[0],dtype=int)) for actor in self._actors]
            all_actions = torch.cat(all_actions, dim=1)
            outs.append(torch.flatten(all_actions[self.n_non_worker_agents:]))
            
            # compute values
            if self.critic_config["model"] == "fc":
                values.append(torch.flatten(self._critic(obss_flat[i])))
            elif self.critic_config["critic_fc"]:
                values.append(torch.flatten(self._critic(x=x, edge_index=fc_edge_index, edge_attr=fc_edge_attr, batch=torch.zeros(x.shape[0],dtype=int))))
            elif not self.critic_config["critic_fc"]:
                values.append(torch.flatten(self._critic(x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr, batch=torch.zeros(x.shape[0],dtype=int))))
       
        # re-batch outputs
        outs = torch.stack(outs)
        self.last_values = torch.stack(values)

        return outs, state
    