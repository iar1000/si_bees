
import random
from ray import tune

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
def filter_actor_gnn_tunables(config):
    config["critic_rounds"] = 1
    return config
