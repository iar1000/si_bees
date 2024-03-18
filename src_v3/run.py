import argparse
import os
import sys
from ray import tune
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from ray.rllib.algorithms.ppo import PPO

from agents import TYPE_MPE_LANDMARK, TYPE_MPE_WORKER, Oracle, Worker
from environment import base_env, load_task_model, marl_env
from task_models import mpe_spread_marl_model, mpe_spread_reduced
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
    import pygame
    from pygame.time import Clock

    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--checkpoint_name",    default=None, help="directory name of run in checkpoints directory")
    parser.add_argument('--checkpoint_nr',      default=-1,   help="number of checkpoint that shall be run, default newest")
    parser.add_argument('--task_level',         default=0,    help="task level of curriculum")
    parser.add_argument('--env_config',         default=None, help="manual set of env config")
    parser.add_argument('--canvas_width',       default=300,  help="set visualisation canvas width")
    parser.add_argument('--canvas_height',      default=300,  help="set visualisation canvas height")
    args = parser.parse_args()

    # load checkpoint
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
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        elif cps:
            cps.sort()
            checkpoint = cps[-1]
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        else:
            checkpoint_path = checkpoint_dir
        
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
    tune.register_env("base_env", lambda _: base_env(config=env_config, initial_task_level=int(args.task_level)))
    tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=int(args.task_level)))
    
    # load policy
    policy_net = None
    if checkpoint_path: policy_net = PPO.from_checkpoint(checkpoint_path)

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

    model = load_task_model(name=env_config["task_model"], env_type=env_type)
    # use pygame to render physical simpulation of mpe
    if model is mpe_spread_marl_model or model is mpe_spread_reduced:
        task = model(config=selected_config,
                      use_cuda=False,
                      inference_mode=True,
                      verbose=True,
                      policy_net=policy_net)
        pygame.init()
        clock = Clock()
        window = pygame.display.set_mode((500, 500))
        font = pygame.font.SysFont('Computer Modern', 24)
        pos_scale = 500.00 / task.grid_size
        landmark_scale = pos_scale
        agent_scale = pos_scale if task.n_workers < 10 else pos_scale * 10
        
        while True:
            # check if quit 
            for event in pygame.event.get() :
                if event.type == pygame.QUIT :
                    pygame.quit()
                    sys.exit()

            # advance model
            task.step()

            # draw
            window.fill((255, 255, 255))
            
            # draw visibility
            # obs = task.get_obs()
            # edges = obs[0][2]
            # froms, tos = list(), list()
            # for j in range(task.n_agents ** 2):
            #     if edges[j][0] == 1: # gym.Discrete(2) maps to one-hot encoding, 0 = [1,0], 1 = [0,1]
            #         froms.append(j // task.n_agents)
            #         tos.append(j % task.n_agents)
            # for f, t in zip(froms, tos):
            #     pygame.draw.line(window, (0, 0, 0, 50), [p * pos_scale for p in task.schedule_all.agents[f].pos], [p * pos_scale for p in task.schedule_all.agents[t].pos], width=1)

            # draw state
            for agent in task.schedule_all.agents:
                if int(agent.type) == int(TYPE_MPE_WORKER):
                    pygame.draw.line(window, (180, 0, 0), [p * pos_scale for p in agent.pos], [p * pos_scale for p in agent.pos + agent.velocity], width=1)
                    pygame.draw.circle(window, (0, 150, 0), [p * pos_scale for p in agent.pos], agent.size * agent_scale, width=0)
                elif int(agent.type) == int(TYPE_MPE_LANDMARK):
                    pygame.draw.circle(window, (150, 0, 0), [p * pos_scale for p in agent.pos], agent.size * landmark_scale, width=1)
            state = font.render(f"t = {int(task.t)}/{int(task.episode_length)}", False, (0, 0, 0))
            window.blit(state, (30,30))
            pygame.display.update()
            clock.tick(1.0/task.dt)

            # automatical restart
            if not task.running:
                task = model(config=selected_config,
                      use_cuda=False,
                      inference_mode=True,
                      policy_net=policy_net)

    # buildin webserver for mesa environment
    else:
        server = ModularServer(
            name=f"{env_config['env_type']} {env_config['task_model']}", 
            model_cls=model, 
            visualization_elements=[canvas], 
            model_params={
                "config": selected_config,
                "inference_mode": True,
                "verbose": True,
                "policy_net": policy_net,
                }
        )
        server.launch(open_browser=True)

 