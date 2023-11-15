from typing import Dict, List
from ray.rllib.utils.framework import TensorType
import torch
from torch.nn import Module, Sequential
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class GNN_base(TorchModelV2, Module):
    """
    base class for one-round gnn models.
    implements following process:
    - encode agents obs -> h_0 = encode(obs)
    - aggregate neighbours hs -> c_0 = aggregator(hs)
    - compute hidden state -> h_1 = f(h_0, c_0)
    - decode to get actions (distribution) -> q ~ decode(h_1)
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
        self.config = self.model_config
        self.custom_config = self.model_config["custom_model_config"]
        self.encoder_config = self.custom_config["encoder"]
        self.aggregator_config = self.custom_config["aggregator"]
        self.f_config = self.custom_config["f"]
        self.decoder_config = self.custom_config["decoder"]
        self.value_config = self.custom_config["value"]

        self.num_inputs = flatdim(obs_space)
        self.num_outputs = num_outputs
        self.state_size = int(self.num_inputs / self.custom_config["n_states"]) 
        self.encoding_size = self.custom_config["encoding_size"]
        self.hidden_state_size = self.custom_config["hidden_state_size"]

        print(f"size obs_space: {self.num_inputs}")
        print(f"size agent state: {self.state_size}")
        print(f"size encoding: {self.encoding_size}")
        print(f"size hidden state: {self.hidden_state_size}")
        print(f"size output: {self.num_outputs}")

        self._encoder = self._build_encoder(self.state_size, self.encoding_size)
        self._aggretator = self._build_aggregator()
        self._f = self._build_f(2 * self.encoding_size, self.hidden_state_size)
        self._decoder = self._build_decoder(self.hidden_state_size, self.num_outputs)
        self._value = self._build_value(self.hidden_state_size, 1)

        print("=== GNN Setup ===")
        print(f"Encoder: {self._encoder}")
        print(f"Aggregator: {self._aggretator}")
        print(f"F: {self._f}")
        print(f"Decoder: {self._decoder}")
        print(f"Value: {self._value}")
        print("")

        # value computed in forward turn and only returned later
        self.last_value_output = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        the obs that an agent returns are the states of himself (at position 0) and the states of all his neighbouring fields.
        the first value is a flag indicating if the corresponding state is valid (1) or not (0), f.e. if no agent is on a neighboring field, state is invalid.
        all states are of the same form.
        f.e. obs_space = Tuple(Tuple(Box(1), Box(3), Box(3)), Tuple(Box(1), Box(3))), BATCH_SIZE=32
            len(obs) = 2 // number of Tuples in the outermost Tuple
            len(obs[0]) = 3 // first of the two nested inner Tuples
            len(obs[1][1]) = 32 // second of the two nested inner Tuples, has batch size 32
            len(obs[1][1][0]) = 3 // second of the two nested inner Tuples, first element of batch has size 3
        """
        obs = input_dict["obs"] # list of lists of nested supspaces of size BATCH_SIZE x obs_subspace_size
        obs_flat = input_dict["obs_flat"] # tensor of shape BATCH_SIZE x flattened obs_space size

        num_states = len(obs)
        batch_size = len(obs_flat)
        """
        one observation consists of many states.
        the first state is the one of the agent itself, should always be valid
            then follow all valid states, e.g. all states of the neighbours
            then follows a zero padding for all invalid, e.g. empty, states
        e.g. obs = s_agent | s_N(agent)_0 | .. | s_N(agent)_m | 0s
        """
        # for each observation, throw away all invalid states and stack up all states together to one batch
        valid_states = list() # holds all valid states
        valid_states_indexes = [0] # tracks index from where to where an obs has states
        for curr_obs in obs_flat:
            # first state is agents own state, always valid
            j = 0
            valid_states.append(curr_obs[0:(j+1) * self.state_size])
            # add all other valid states of this observation to batch for encoding
            while j+1 < num_states and curr_obs[(j+1) * self.state_size] == 1:
                valid_states.append(curr_obs[(j+1) * self.state_size:(j+2) * self.state_size])
                j += 1
            valid_states_indexes.append(valid_states_indexes[-1] + j+1)
        valid_states = torch.stack(valid_states)

        # encode all states as one huge batch of size NUM_VALID_STATES_IN_BATCH x state_size
        encoded_states = self._encoder(valid_states)

        # stich back together the states to the corresponding observation i
        agent_encodings = list() # agent_states[i] holds encoded state of agent in observation i
        neighbors_encodings = list() # neighbors_states[i] holds list of encoded states of neighbors of agent in observation i
        for i in range(batch_size):
            from_index = valid_states_indexes[i]
            to_index = valid_states_indexes[i+1]

            agent_encodings.append(encoded_states[from_index]) # get agents encoded state
            neighbors_encodings.append([])
            # add neighbors encoded states or zero vector if none
            if to_index - from_index > 1:
                for j in range(from_index+1, to_index):
                    neighbors_encodings[-1].append(encoded_states[j])
            else:
                neighbors_encodings[-1].append(torch.zeros_like(encoded_states[from_index]))

        # aggretate the states of the neighbors
        neighbors_aggregations = list()
        for nhe in neighbors_encodings:
            neighbors_aggregations.append(self._aggretator(nhe))
        
        # concat agent encoded state and aggregated states of neighbors to get again a batch of size BATCH_SIZE x 2 * encoding_size
        f_in = list()
        for j in range(batch_size):
            f_in.append(torch.cat((agent_encodings[j], neighbors_aggregations[j])))
        f_in = torch.stack(f_in)

        # run it through f
        hidden_state = self._f(f_in)

        self.last_value_output = self._value(hidden_state)
        return self._decoder(hidden_state), state
    
    def value_function(self):
        return torch.reshape(self.last_value_output, [-1])
    
    def _build_encoder(self, input_size: int, output_size: int) -> Module:
        """creates an encoder that takes the state of one agent and returns an encoding of it"""
        raise NotImplementedError
        
    def _build_aggregator(self) -> Module:
        """builds aggregator function that takes in a list of encoded states and aggregates them"""
        raise NotImplementedError
        
    def _build_f(self, input_size: int, output_size: int) -> Module:
        """builds network that takes encoded state of calling agent and aggregated encoded states of neighbours and computes hidden state"""
        raise NotImplementedError

    def _build_decoder(self, input_size: int, output_size: int) -> Module:
        """builds decoder network that takes the hidden state and outputs the action distribution"""
        raise NotImplementedError

    def _build_value(self, input_size: int, output_size: int) -> Module:
        """builds decoder network that takes the hidden state and outputs the value function"""
        raise NotImplementedError

class GNN_ComNet(GNN_base):

    def _build_encoder(self, input_size: int, output_size: int) -> Module:
        """expects a dict in form:
                hidden_layers: [size_hidden_layer_0, ..., size_hidden_layer_n],
                activation: name of pytorch activation function to us
        """
        layers = list()
        prev_layer_size = input_size
        for curr_layer_size in self.encoder_config["hidden_layers"]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.encoder_config["activation"]))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=output_size, activation_fn=self.encoder_config["activation"]))

        return Sequential(*layers)
    
    def _build_aggregator(self) -> Module:
        """expects a dict in form:
                ops: {sum | mean}
        """
        # if there is not enough neighbors, skip aggretation
        def trivial_case_wrapper(func, hs):
            if len(hs) < 2:
                return hs[0]
            return func(hs)
        
        def mean(hs):
            return torch.mean(torch.stack(hs), dim=0)
        
        def sum(hs):
            return torch.sum(torch.stack(hs), dim=0)

        if self.aggregator_config["op"] == "mean":
            return lambda hs: trivial_case_wrapper(mean, hs) 
        if self.aggregator_config["op"] == "sum":
            return lambda hs: trivial_case_wrapper(sum, hs)       
    
    def _build_f(self, input_size: int, output_size: int) -> Module:
        """expects a dict in form:
                hidden_layers: [size_hidden_layer_0, ..., size_hidden_layer_n],
                activation: name of pytorch activation function to us
        """
        layers = list()
        prev_layer_size = input_size
        for curr_layer_size in self.f_config["hidden_layers"]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.f_config["activation"]))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=output_size, activation_fn=self.f_config["activation"]))

        return Sequential(*layers)
    
    def _build_decoder(self, input_size: int, output_size: int) -> Module:
        """expects a dict in form:
                hidden_layers: [size_hidden_layer_0, ..., size_hidden_layer_n],
                activation: name of pytorch activation function to us
        """
        layers = list()
        prev_layer_size = input_size
        for curr_layer_size in self.decoder_config["hidden_layers"]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.decoder_config["activation"]))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=output_size, activation_fn=self.decoder_config["activation"]))

        return Sequential(*layers)
    
    def _build_value(self, input_size: int, output_size: int) -> Module:
        """expects a dict in form:
                hidden_layers: [size_hidden_layer_0, ..., size_hidden_layer_n],
                activation: name of pytorch activation function to us
        """
        layers = list()
        prev_layer_size = input_size
        for curr_layer_size in self.value_config["hidden_layers"]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.value_config["activation"]))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=output_size, activation_fn=self.value_config["activation"]))

        return Sequential(*layers)