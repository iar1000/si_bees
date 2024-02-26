import argparse
import os
from ray import tune
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from ray.rllib.algorithms.ppo import PPO

from agents import Oracle, Worker
from environment import base_env, load_task_model, marl_env
from utils import read_yaml_config

def agent_visualisation(agent):
    colors = ["green", "black", "red", "blue", "orange", "yellow"]
    if agent is None:
        return
    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.9, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 1}
    if type(agent) is Oracle:
        return {"Shape": "rect", "w": 1, "h": 1, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--checkpoint_name",    default=None, help="directory name of run in checkpoints directory")
    parser.add_argument('--checkpoint_nr',      default=-1,   help="number of checkpoint that shall be run, default newest")
    parser.add_argument('--task_level',         default=0,    help="task level of curriculum")
    parser.add_argument('--env_config',         default=None, help="manual set of env config")
    parser.add_argument('--canvas_width',       default=300,  help="set visualisation canvas width")
    parser.add_argument('--canvas_height',      default=300,  help="set visualisation canvas height")
    args = parser.parse_args()

    # load policy
    policy_net = None
    checkpoint_dir = os.path.join("checkpoints", args.checkpoint_name) if args.checkpoint_name else None
    checkpoint_path = None
    if checkpoint_dir:
        # get all checkpoints
        env_config = [d for d in os.listdir(checkpoint_dir) if "env_config" in d][0]
        cps = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d)) and "checkpoint" in d]
        # select checkpoint number
        if int(args.checkpoint_nr) > 0:
            options = [cp for cp in cps if args.checkpoint_nr in cp]
            options.sort()
            checkpoint = options[0]
        else:
            cps.sort()
            checkpoint = cps[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        policy_net = PPO.from_checkpoint(checkpoint_path)
        
    # load task config from checkpoint dir or from src3/configs if env_config is specified
    env_config_path = os.path.join(checkpoint_dir, env_config) if checkpoint_dir \
        else os.path.join("src_v3", "configs", args.env_config) if args.env_config \
        else None
    assert env_config_path, "must load a checkpoint, specifiy env_path or both to make this happen"
    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[int(args.task_level)]

    # register environments
    assert env_config["env_type"] in {"rl", "marl"}, f"env_type {env_config['env_type']} is not supported"
    env_type = base_env(env_config) if env_config["env_type"] == "rl" else marl_env(env_config)
    tune.register_env("base_env", lambda env_config: base_env(config=env_config))
    tune.register_env("marl_env", lambda env_config: marl_env(config=env_config))

    # create visualisation canvas
    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=selected_config["model"]["grid_size"], 
        grid_height=selected_config["model"]["grid_size"], 
        canvas_width=int(args.canvas_width),
        canvas_height=int(args.canvas_height))
    
    # create and launch
    print("\n=========== LAUNCH RUN =============")
    print("checkpoint_name  = ", args.checkpoint_name)
    print("checkpoint_nr    = ", int(args.checkpoint_nr))
    print("checkpoint_path  = ", checkpoint_path)
    print("env_config_path  = ", env_config_path)
    print("task level       = ", args.task_level)
    print("\n\n")

    server = ModularServer(
        name=f"{env_config['env_type']} {env_config['task_model']}", 
        model_cls=load_task_model(name=env_config["task_model"], env_type=env_type), 
        visualization_elements=[canvas], 
        model_params={
            "config": selected_config,
            "inference_mode": True,
            "policy_net": policy_net,
            }
    )
    server.launch(open_browser=True)

 