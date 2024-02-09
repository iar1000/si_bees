import argparse
import os
import os
from ray import tune
from mesa.visualization.ModularVisualization import ModularServer, TextElement
from mesa.visualization.modules import CanvasGrid
from ray.rllib.algorithms.ppo import PPO

from agents_marl import Oracle, Worker
from model_marl import get_model_by_config
from environment_marl import Marl_env
from utils import read_yaml_config

class GamestateTextElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        out = [
            f"states                = {model.oracle.output} {[a.output for a in model.schedule_workers.agents]}",
            f"",
            f"reward_total          = {model.reward_total}",
            f"reward_lower_bound    = {model.reward_lower_bound}",
            f"reward_upper_bound    = {model.reward_upper_bound}",
            f"reward_percentile     = {(model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if (model.reward_upper_bound - model.reward_lower_bound) != 0 else 0}",
            f"",
            f"state_switch_pause    = {model.ts_curr_state}/{model.state_switch_pause}",
            f"n_state_switches      = {model.n_state_switches}",
        ]
        return "<h3>Status</h3>" + "<br />".join(out)
 
def agent_visualisation(agent):
    colors = ["green", "black", "red", "blue", "orange", "yellow"]
    if agent is None:
        return
    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.9, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 1}
    if type(agent) is Oracle:
        return {"Shape": "rect", "w": 1, "h": 1, "Color": colors[agent.output % 6], "Filled": "true", "Layer": 0}
    
def create_server(model_checkpoint: str, env_config_path: str, env_config: str, task_level: int):

    config = read_yaml_config(env_config_path)
    curriculum_configs = [config[task] for task in config]
    task_config = curriculum_configs[task_level]
    model = get_model_by_config(env_config)

    tune.register_env("Marl_env", lambda _: Marl_env(config=config, env_config_file=env_config, initial_task_level=task_level))

    canvas = CanvasGrid(
        agent_visualisation, 
        grid_width=task_config["model"]["grid_size"], 
        grid_height=task_config["model"]["grid_size"], 
        canvas_width=300,
        canvas_height=300)
    
    game_state = GamestateTextElement()

    server = ModularServer(
        model, 
        [canvas, game_state], 
        "swarm intelligence with graph networks", 
        model_params={
            "config": task_config,
            "inference_mode": True,
            "policy_net": PPO.from_checkpoint(model_checkpoint) if model_checkpoint else None},
    )

    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--run_dir",            default=None, help="directory name of run in checkpoints directory")
    parser.add_argument('--checkpoint_nr',      default=-1,   help="number of checkpoint that shall be run, default newest")
    parser.add_argument('--task_level',         default=0,    help="task level of curriculum")
    parser.add_argument('--env_config',         default=None, help="manual set of env config")
    args = parser.parse_args()

    run_dir = os.path.join("checkpoints", args.run_dir) if args.run_dir else None
    if run_dir:
        env_config = [d for d in os.listdir(run_dir) if "env_config" in d][0]
        cps = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and "checkpoint" in d]
        if int(args.checkpoint_nr) > 0:
            options = [cp for cp in cps if args.checkpoint_nr in cp]
            options.sort()
            checkpoint = options[0]
        else:
            cps.sort()
            checkpoint = cps[-1]

    if args.env_config:
        env_config = args.env_config

    # final paths
    checkpoint_path = os.path.join(run_dir, checkpoint) if run_dir else None
    config_path = os.path.join(run_dir, env_config) if run_dir \
        else os.path.join("src_v2", "configs", env_config) if args.env_config \
            else None
    
    print("\n\n=========== LAUNCH RUN =============")
    print("checkpoint_path  = ", checkpoint_path)
    print("config_path      = ", config_path)
    print("task level       = ", args.task_level)
    print("\n\n")

    # launch
    server = create_server(model_checkpoint=checkpoint_path,
                           env_config_path=config_path, 
                           env_config=env_config, 
                           task_level=int(args.task_level))
    server.launch(open_browser=True)

 