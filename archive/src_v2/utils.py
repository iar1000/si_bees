
import random
from ray import tune
import torch
from gymnasium.spaces import Tuple
import yaml


def get_relative_pos(p1, p2) -> [int, int]:
    """
    returns relative position of two points
    """
    x1, y1 = p1
    x2, y2 = p2
    relative_x = x2 - x1
    relative_y = y2 - y1
    return (relative_x, relative_y)

def relative_moore_to_linear(p, radius):
    """
    compute index of a point in a moore neighborhood
    indexes the neighborhood on a row-by-row basis
    """
    x, y = p
    x_shifted = x + radius
    y_shifted = y + radius
    linear_index = y_shifted * (2 * radius + 1) + x_shifted
    return linear_index

def get_random_pos_on_border(center, dist: int):
    """returns coordinates of a point on the border dist away from the center"""
    center_x, center_y = center
    
    side = random.randint(0,3)
    pos_x, pos_y = center_x, center_y
    # up
    if side == 0:
        pos_y = center_y + dist
        pos_x = random.randint(center_x - dist, center_x + dist)
    # right
    elif side == 1:
        pos_x = center_x + dist
        pos_y = random.randint(center_y - dist, center_y + dist)
    # down
    elif side == 2:
        pos_y = center_y - dist
        pos_x = random.randint(center_x - dist, center_x + dist)    
    # left
    else:
        pos_x = center_x - dist
        pos_y = random.randint(center_y - dist, center_y + dist)

    return (pos_x, pos_y)

# create internal model from config
def create_tunable_config(config):
    tunable_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            if isinstance(v["min"], int) and isinstance(v["max"], int):
                tunable_config[k] = tune.choice(list(range(v["min"], v["max"] + 1)))
            else:
                tunable_config[k] = tune.uniform(v["min"], v["max"])       
        elif isinstance(v, list):
            tunable_config[k] = tune.choice(v)
        else:
            tunable_config[k] = v
    return tunable_config

# set num rounds of actor config to one, as being overriden in a later stage
def filter_tunables(config):
    config["rounds"] = 1
    return config

def read_yaml_config(path: str):
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File not found: {path}")

# builds graph from observation
def build_graph_v2(num_agents: int, agent_obss: Tuple, edge_obss: Tuple, batch_index: int):
    x = []
    # concatenate all agent observations into a single tensor
    for j in range(num_agents):
        curr_agent_obs = torch.cat(agent_obss[j], dim=1)
        x.append(curr_agent_obs[batch_index])

    # build edge index from adjacency matrix
    actor_froms, actor_tos, actor_edge_attr = [], [], []
    fc_froms, fc_tos, fc_edge_attr = [], [], []
    for j in range(num_agents ** 2):
        curr_edge_obs = torch.cat(edge_obss[j], dim=1)
        
        # add edge to actor graph
        if curr_edge_obs[0][1] == 1: # gym.Discrete(2) maps to one-hot encoding, 0 = [1,0], 1 = [0,1]
            actor_froms.append(j // num_agents)
            actor_tos.append(j % num_agents)
            actor_edge_attr.append(curr_edge_obs[batch_index])
        # add edge to fc graph
        fc_froms.append(j // num_agents)
        fc_tos.append(j % num_agents)
        fc_edge_attr.append(curr_edge_obs[batch_index])

    return x, [actor_froms, actor_tos], actor_edge_attr, [fc_froms, fc_tos], fc_edge_attr

def get_active_agents(agent_obss: Tuple) -> int:
    """determine which agent has active flag. depends on the obs space"""
    mapping = {}
    for j in range(len(agent_obss)):
        curr_agent_obs = torch.cat(agent_obss[j], dim=1)
        for i in range(len(curr_agent_obs)):
            if curr_agent_obs[i][1] == 1:
                mapping[i] = j
    return mapping

def compute_agent_placement(num_workers: int, communication_range: int,
                            grid_size: int, 
                            oracle_pos: tuple, placement_strategy: str):
    agent_positions = list()
    direction_vector = random.choice([[1,0], [-1,0], [0,1], [0,-1]])
    curr_pos = oracle_pos
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
    
    # check that all agents are in bounds
    for i, (x, y) in enumerate(agent_positions):
        # x_new = x - grid_size if x > grid_size else x
        # y_new = y - grid_size if y > grid_size else y
        # agent_positions[i] = (x_new, y_new)
        middle = grid_size / 2
        if x >= grid_size or x < 0 or y >= grid_size or y < 0:
            agent_positions[i] = (random.randint(max(middle - 5, 0), min(middle + 5, grid_size-1)), random.randint(max(middle - 5, 0), min(middle + 5, grid_size-1)))

    return agent_positions
            

