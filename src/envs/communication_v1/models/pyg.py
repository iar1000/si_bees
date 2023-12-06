from typing import Dict, List
import numpy as np
import torch
from torch import TensorType
from torch.nn import Module, Sequential, Identity
from torch_geometric.nn import Sequential as PyG_Sequential, global_mean_pool
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

from envs.communication_v1.model import MAX_COMMUNICATION_RANGE
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
        self.n_non_worker_agents = 2

        # todo: check if actor, critic and encoder configs are speciyin supported models
        assert self.actor_config["model"] in SUPPORTED_ACTORS, f"actor model {self.actor_config['model']} not supported"
        assert self.critic_config["model"] in SUPPORTED_CRITICS, f"critic model {self.critic_config['model']} not supported"
        assert self.encoders_config["node_encoder"] in SUPPORTED_ENCODERS, f"node encoder model {self.encoders_config['node_encoder']} not supported"
        assert self.encoders_config["edge_encoder"] in SUPPORTED_ENCODERS, f"edge encoder model {self.encoders_config['edge_encoder']} not supported"

        self.num_inputs = flatdim(obs_space.original_space)
        self.num_agents = len(obs_space.original_space[0])
        self.num_outputs = num_outputs
        self.per_agent_outputs = num_outputs // (self.num_agents - self.n_non_worker_agents) 
        self.adj_matrix_size = len(obs_space.original_space[1])
        self.node_state_size = flatdim(obs_space.original_space[0][0])
        self.edge_state_size = flatdim(obs_space.original_space[1][0])
        
        self.last_values = None
        self._node_encoder, self._edge_encoder, self.encoding_size = self._build_encoders(
            config=self.encoders_config, 
            node_state_size=self.node_state_size,
            edge_state_size=self.edge_state_size)
        self._actor = self.__build_gnn(
            config=self.actor_config, 
            ins=self.encoding_size, 
            outs=self.per_agent_outputs,
            edge_dim=self.encoding_size)
        self._critic = self._build_critic(
            config=self.critic_config, 
            ins=self.num_inputs if self.critic_config["model"] == "fc" else self.encoding_size,
            edge_dim=self.encoding_size)
        
        # build sincos helper for edge encoder
        if self.encoders_config["edge_encoder"] == "sincos":
            self.pos_encoding_size = self.encoding_size // 2 # concat x and y positional encodings, e.g. each pos_encoding only half as big as encoding_size
            max_sequnce_length = 2 * MAX_COMMUNICATION_RANGE + 1
            angle_lookup = [1 / (10000 ** (2 * i / self.pos_encoding_size)) for i in range(self.pos_encoding_size // 2)]
            self.sin_encodings = [[np.sin(seq_num * angle_lookup[i]) for seq_num in range(max_sequnce_length)] for i in range(self.pos_encoding_size // 2)]
            self.cos_encodings = [[np.cos(seq_num * angle_lookup[i]) for seq_num in range(max_sequnce_length)] for i in range(self.pos_encoding_size // 2)]
            self.rel_to_sequence = lambda relative_position: relative_position + MAX_COMMUNICATION_RANGE

        
        print("\n=== backend model ===")
        print(f"num_agents        = {self.num_agents}")
        print(f"num_inputs        = {self.num_inputs}")
        print(f"num_outputs       = {self.num_outputs}")
        print(f"per_agent_outputs = {self.per_agent_outputs}")
        print(f"adj_matrix_size   = {self.adj_matrix_size}")
        print(f"node_state_size   = {self.node_state_size}")
        print(f"edge_state_size   = {self.edge_state_size}")
        print(f"encoding size     = {self.encoding_size}")
        print("actor: ", self.actor_config)
        print("critic: ", self.critic_config)
        print("encoders: ", self.encoders_config)
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

    def _build_critic(self, config: dict, ins: int, edge_dim: int) -> Module:
        """creates an NN model based on the config dict"""
        outs = 1

        if config["model"] != "fc":
            # if used as critic, can use multiple message passing rounds, before flatten all agents outputs and connect them to a single output
            # note: hidden_size = encoding_size, because output of one round is used as input for the next round, while shape of edge_attr is not changed 
            gnn_rounds = list()
            for _ in range(config["critic_rounds"] - 1):
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
                gnn_rounds.append(torch.nn.ReLU(inplace=True))
            gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
            gnn_rounds.append((global_mean_pool, 'x, batch -> x'))
            gnn_rounds.append(SlimFC(in_size=ins, out_size=outs))

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
            # encode node values from agent observations
            x = []
            for j in range(self.num_agents):
                # concatenate all agent observations into a single tensor and encode them
                curr_agent_obs = torch.cat(agent_obss[j], dim=1)
                x.append(self._node_encoder(curr_agent_obs[i]))
            x = torch.stack(x)

            # build edge index from adjacency matrix
            froms = []
            tos = []
            edge_attr = []
            for j in range(self.adj_matrix_size):
                is_active = edge_obss[j][0][i][1] == 1 # gym.Discrete(2) maps to one-hot encoding, 0 = [1,0], 1 = [0,1]
                curr_edge_obs = torch.cat(edge_obss[j], dim=1)
                if is_active: 
                    froms.append(j // self.num_agents)
                    tos.append(j % self.num_agents)
                    edge_attr.append(self._edge_encoder(curr_edge_obs[i]))
            edge_index = torch.tensor([froms, tos], dtype=torch.int64)
            if edge_attr:
                edge_attr = torch.stack(edge_attr)
            else:
                edge_attr = torch.zeros((0, self.encoding_size), dtype=torch.float32)

            # compute actions
            all_actions = self._actor(x=x, edge_index=edge_index, edge_attr=edge_attr)
            outs.append(torch.flatten(all_actions[self.n_non_worker_agents:]))
            
            # compute values
            if self.critic_config["model"] == "fc":
                values.append(torch.flatten(self._critic(obss_flat[i])))
            else:
                values.append(torch.flatten(self._critic(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=torch.zeros(x.shape[0],dtype=int))))
       
        # re-batch outputs
        outs = torch.stack(outs)
        self.last_values = torch.stack(values)

        return outs, state
    