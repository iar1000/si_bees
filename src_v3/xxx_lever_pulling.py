import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config


def evaluate_lever_pulling(name: str, checkpoint_path: str, env_config_path: str, 
                             task_level: int = 0, eval_episodes: int = 50):

    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]
    
    tune.register_env("marl_env", lambda _: marl_env(config=env_config))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path)
    
    ratios = list()
    for _ in range(eval_episodes):
        model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
        _, rewards, _, _ = model.step()
        ratios.append(rewards[0])

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    print(means, stdevs)

    with open(f"/Users/sega/Code/si_bees/reports/report_lever_pulling_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"mean: {means}\n")
        file.write(f"stdev: {stdevs}\n")