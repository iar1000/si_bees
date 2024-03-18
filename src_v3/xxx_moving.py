import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from tqdm import tqdm
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config

def evaluate_moving_MARL(name: str, checkpoint_path: str, env_config_path: str, 
                         task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)

    env_config = read_yaml_config(os.path.join("reports", "env_config_moving_marl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=i))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None

    means = list()
    stdevs = list()
    state_switchess = list()


    ratios = list()
    state_switches = list()

    files = ["num_connect", "connect_correct", "avg_dist", "avg_dist_connect", "avg_dist_correct"]
    with tqdm(total=eval_episodes, desc="episodes") as pbar:
        for k in range(eval_episodes):
            data = [list() for _ in files]

            model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
            done = False
            while not done:
                _, _, _, truncated = model.step()
                done = truncated["__all__"]
                ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
                
                for j, v in enumerate([model.last_number_connected, model.last_connected_correct, model.last_average_distance, model.last_average_distance_connected, model.last_average_distance_correct]):
                    data[j].append(v)
            
            # save stats
            for j, f in enumerate(files):
                with open(f"reports/report_moving_task–name_{task_level}_{f}.csv", "a+") as file:
                    file.write(', '.join([str(k)] + [str(d) for d in data[j]])) 
                    file.write("\n")  
            state_switches.append(model.n_state_switches)
        
            # Update progress bar
            pbar.update(1)

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    state_switchess = round(mean(state_switches))
    print(means, stdevs, state_switchess)

    with open(f"/Users/sega/Code/si_bees/reports/report_moving_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"{means} $\pm$ {stdevs}, {state_switches}")

def evaluate_moving_history_MARL(name: str, checkpoint_path: str, env_config_path: str, 
                         task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)

    env_config = read_yaml_config(os.path.join("reports", "env_config_moving_history_marl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=i))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None

    means = list()
    stdevs = list()
    state_switchess = list()


    ratios = list()
    state_switches = list()

    files = ["num_connect", "connect_correct", "avg_dist", "avg_dist_connect", "avg_dist_correct"]
    with tqdm(total=eval_episodes, desc="episodes") as pbar:
        for k in range(eval_episodes):
            data = [list() for _ in files]

            model = task(config=selected_config,
                            use_cuda=False,
                            inference_mode=True,
                            policy_net=policy_net)
            done = False
            while not done:
                _, _, _, truncated = model.step()
                done = truncated["__all__"]
                ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
                
                for j, v in enumerate([model.last_number_connected, model.last_connected_correct, model.last_average_distance, model.last_average_distance_connected, model.last_average_distance_correct]):
                    data[j].append(v)
            
            # save stats
            for j, f in enumerate(files):
                with open(f"reports/report_moving_history_task–name_{task_level}_{f}.csv", "a+") as file:
                    file.write(', '.join([str(k)] + [str(d) for d in data[j]])) 
                    file.write("\n")  
            state_switches.append(model.n_state_switches)
        
            # Update progress bar
            pbar.update(1)

    means = round(mean(ratios), 2)
    stdevs = round(stdev(ratios), 2)
    state_switchess = round(mean(state_switches))
    print(means, stdevs, state_switchess)

    with open(f"/Users/sega/Code/si_bees/reports/report_moving_history_{name}_{task_level}.tex", "w") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"{means} $\pm$ {stdevs}, {state_switches}")
