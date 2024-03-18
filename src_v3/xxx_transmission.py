import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from tqdm import tqdm
from environment import MARL_ENV, RL_ENV, base_env, load_task_model, marl_env
from utils import read_yaml_config

def evaluate_transmission_RL(name: str, checkpoint_path: str, env_config_path: str, 
                             task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)

    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("base_env", lambda _: base_env(config=env_config, initial_task_level=task_level))
    task = load_task_model(name=env_config["task_model"], env_type=RL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None
    
    ratios = list()
    state_switches = list()
    with tqdm(total=eval_episodes, desc="episodes") as pbar:
        for _ in range(eval_episodes):
            model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
            done = False
            while not done:
                _, _, _, truncated = model.step()
                done = truncated
                ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
            state_switches.append(model.n_state_switches)
            
            # Update progress bar
            pbar.update(1)

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    state_switchess = round(mean(state_switches), 2)
    print(means, stdevs, state_switchess)

    with open(f"/Users/sega/Code/si_bees/reports/report_transmission_rl_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"{means} - {stdevs} - {state_switchess}")

def evaluate_transmission_MARL(name: str, checkpoint_path: str, env_config_path: str, 
                             task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)

    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("base_env", lambda _: base_env(config=env_config, initial_task_level=task_level))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None
    
    ratios = list()
    state_switches = list()
    with tqdm(total=eval_episodes, desc="episodes") as pbar:
        for _ in range(eval_episodes):
            model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
            done = False
            while not done:
                _, _, _, truncated = model.step()
                done = truncated
                ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
            state_switches.append(model.n_state_switches)
            
            # Update progress bar
            pbar.update(1)

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    state_switchess = round(mean(state_switches), 2)
    print(means, stdevs, state_switchess)

    with open(f"/Users/sega/Code/si_bees/reports/report_transmission_marl_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"{means} $\pm$ {stdevs}, {state_switches}\n")


def evaluate_transmission_extended_MARL(name: str, checkpoint_path: str, env_config_path: str, 
                             task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)

    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("base_env", lambda _: base_env(config=env_config, initial_task_level=task_level))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None
    
    ratios = list()
    state_switches = list()
    with tqdm(total=eval_episodes, desc="episodes") as pbar:
        for _ in range(eval_episodes):
            model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
            done = False
            while not done:
                _, _, _, truncated = model.step()
                done = truncated
                ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
            state_switches.append(model.n_state_switches)
            
            # Update progress bar
            pbar.update(1)

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    state_switchess = round(mean(state_switches), 2)
    print(means, stdevs, state_switchess)

    with open(f"/Users/sega/Code/si_bees/reports/report_transmission_ext_marl_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"{means} $\pm$ {stdevs}, {state_switches}\n")