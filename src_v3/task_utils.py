
import random
from gymnasium.spaces import Tuple
import torch

def get_relative_pos(p1, p2):
    """returns relative position of two points"""
    x1, y1 = p1
    x2, y2 = p2
    relative_x = x2 - x1
    relative_y = y2 - y1
    return (relative_x, relative_y)

def get_worker_outputs(n_workers: int,
                       n_oracle_states: int,
                       oracle_output: int,
                       init_strategy: str):
    """compute list of worker outputs based on initialisation strategy"""
    if init_strategy == "uniform":
            r = random.randint(0, n_oracle_states-1)
            while r == oracle_output:
                r = random.randint(0, n_oracle_states-1)
            return [r for _ in range(n_workers)]
    else:
        return [random.randint(0, n_oracle_states-1) for _ in range(n_workers)]

def get_worker_placements(num_workers: int, 
                            communication_range: int,
                            grid_size: int, 
                            oracle_pos: tuple, 
                            placement_strategy: str) -> list:
    """compute list of worker locations based on placement strategy, randomly distribute out of bounds agents within the grid"""
    agent_positions = list()
    direction_vector = random.choice([[1,0], [-1,0], [0,1], [0,-1]])
    curr_pos = oracle_pos

    # compute worker locations 
    for _ in range(num_workers):
            x_old, y_old = curr_pos

            if placement_strategy == "line_unidirectional":
                curr_pos = x_old + 1, y_old
            elif placement_strategy == "line_multidirectional":
                curr_pos = x_old + direction_vector[0], y_old + direction_vector[1]
            elif placement_strategy == "random_multidirectional":
                if direction_vector[0]:
                    curr_pos = x_old + direction_vector[0] * random.randint(1, max(1, communication_range - 1)), y_old + random.randint(-communication_range+1, communication_range-1)
                else:
                    curr_pos = x_old + random.randint(-communication_range+1, communication_range-1), y_old + direction_vector[1] * random.randint(1, max(1, communication_range - 1))
            else:
                curr_pos = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

            agent_positions.append(curr_pos)
    
    # randomly distribute out-of-bounds agents within a 5-field radius around the oracle
    for i, (x, y) in enumerate(agent_positions):
        middle = grid_size / 2
        if x >= grid_size or x < 0 or y >= grid_size or y < 0:
            agent_positions[i] = (random.randint(max(middle - 5, 0), min(middle + 5, grid_size-1)), random.randint(max(middle - 5, 0), min(middle + 5, grid_size-1)))

    return agent_positions
        
def get_graph_from_batch_obs(num_agents: int, 
                         agent_obss: Tuple, 
                         edge_obss: Tuple, 
                         batch_index: int):
    """
    get graph data for sample "batch_index" of the batched observations
    returns data for graph build from active edges and for fully connected graph
    """
    x = []
    # concatenate all agent observations into a single tensor
    # the first two numbers are not part of the state, they only indicate the active agent
    for j in range(num_agents):
        curr_agent_obs = torch.cat(agent_obss[j], dim=1)
        x.append(curr_agent_obs[batch_index][2:])

    # build edge index from adjacency matrix
    actor_froms, actor_tos, actor_edge_attr = [], [], []
    fc_froms, fc_tos, fc_edge_attr = [], [], []
    # the first two numbers are not part of the state, they only indicate the visibility of the edge
    for j in range(num_agents ** 2):
        curr_edge_obs = torch.cat(edge_obss[j], dim=1)
        
        # add edge to actor graph
        if curr_edge_obs[0][1] == 1: # gym.Discrete(2) maps to one-hot encoding, 0 = [1,0], 1 = [0,1]
            actor_froms.append(j // num_agents)
            actor_tos.append(j % num_agents)
            actor_edge_attr.append(curr_edge_obs[batch_index][2:])
        # add edge to fc graph
        fc_froms.append(j // num_agents)
        fc_tos.append(j % num_agents)
        fc_edge_attr.append(curr_edge_obs[batch_index][2:])

    return x, [actor_froms, actor_tos], actor_edge_attr, [fc_froms, fc_tos], fc_edge_attr