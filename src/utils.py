
import random
from ray import tune
import torch
from gymnasium.spaces import Space, Box, Tuple, Discrete


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

# builds graph from observation
def build_graph_v2(num_agents: int, agent_obss: Tuple, edge_obss: Tuple, batch_index: int):
    # check if data is preprocessed or not
    try:
        curr_edge_obs = torch.cat(edge_obss[0], dim=1)
        distribution_data = True
    except IndexError:
        distribution_data = False

    x = []
    actor_froms, actor_tos, actor_edge_attr = [], [], []
    fc_froms, fc_tos, fc_edge_attr = [], [], []
    
    if distribution_data:
        # concatenate all agent observations into a single tensor
        for j in range(num_agents):
            curr_agent_obs = torch.cat(agent_obss[j], dim=1)
            x.append(curr_agent_obs[batch_index])

        # build edge index from adjacency matrix
        for j in range(num_agents ** 2):
            curr_edge_obs = torch.cat(edge_obss[j], dim=1)
            
            # add edge to actor graph
            if edge_obss[j][0][batch_index][1] == 1: # gym.Discrete(2) maps to one-hot encoding, 0 = [1,0], 1 = [0,1]
                actor_froms.append(j // num_agents)
                actor_tos.append(j % num_agents)
                actor_edge_attr.append(curr_edge_obs[batch_index])
            # add edge to fc graph
            fc_froms.append(j // num_agents)
            fc_tos.append(j % num_agents)
            fc_edge_attr.append(curr_edge_obs[batch_index])
       # print(x, actor_edge_attr)
    else:
        # concatenate all agent observations into a single tensor
        for j in range(num_agents):
            curr_agent_obs = torch.cat(agent_obss[j], dim=0)
            x.append(curr_agent_obs[batch_index])

        # build edge index from adjacency matrix
        for j in range(num_agents ** 2):
            curr_edge_obs = torch.cat([torch.flatten(t) for t in edge_obss[j]]) # weird nesting
            # add edge to actor graph
            if edge_obss[j][0][batch_index] == 1: # actual number, 1 if valid graph
                actor_froms.append(j // num_agents)
                actor_tos.append(j % num_agents)
                actor_edge_attr.append(curr_edge_obs[batch_index])
            # add edge to fc graph
            fc_froms.append(j // num_agents)
            fc_tos.append(j % num_agents)
            fc_edge_attr.append(curr_edge_obs[batch_index])
        #print(x, actor_edge_attr)
        
    return x, [actor_froms, actor_tos], actor_edge_attr, [fc_froms, fc_tos], fc_edge_attr