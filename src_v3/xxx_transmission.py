import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from tqdm import tqdm
from environment import MARL_ENV, RL_ENV, base_env, load_task_model, marl_env
from utils import read_yaml_config

EVAL_EPISODES = 20

def evaluate_transmission_RL(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_transmission_rl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    
    # 250k step trainingh with ASHA, 300 Trials
    checkpoints = [
        # binary
        os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-32-r162"),                        # GINE
        os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-13-r64"),                         # GATv2
        os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-46-r222"),                        # Transformer
        # sum
        os.path.join("checkpoints", "20240228_transmission_rl_2_14-58-21-r19"),                         # GINE
        os.path.join("checkpoints", "20240226_transmission_rl_2_19-00-41-r169", "checkpoint_000000"),   # GATv2
        os.path.join("checkpoints", "20240226_transmission_rl_2_19-01-05-r268"),                        # Transformer
        # baseline
        None,
    ]

    # run all curriculum levels
    for i, selected_config in enumerate(task_configs):
        if exclude and i in exclude: continue

        print(f"\n\n\n########### run level {i} ##############\n\n\n")

        tune.register_env("base_env", lambda _: base_env(config=env_config, initial_task_level=i))
        task = load_task_model(name=env_config["task_model"], env_type=RL_ENV)
        
        means = list()
        stdevs = list()
        state_switchess = list()

        for checkpoint in checkpoints:
            policy_net = PPO.from_checkpoint(checkpoint) if checkpoint else None
            print(f"run {checkpoint} level {i}")

            ratios = list()
            state_switches = list()

            # Initialize progress bar for iterations
            with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
                for _ in range(EVAL_EPISODES):
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

            means.append(round(mean(ratios), 2))
            stdevs.append(round(stdev(ratios), 2))
            state_switchess.append(round(mean(state_switches), 2))
            print(means, stdevs, state_switchess)

        print(means, stdevs, state_switchess)

        table = f"""
    \\begin{{table}}[h]
    \centering
    \\begin{{tabular}}{{|c|c|c|c|}}
    \hline
    \\textbf{{Convolution}} & \\textbf{{Reward}} & \\textbf{{Result}} \\\\
    \hline
    GINE & shared binary & {means[0]} $\pm$ {stdevs[0]} \\\\
    GATv2 & shared binary & {means[1]} $\pm$ {stdevs[1]} \\\\
    Transformer & shared binary & {means[2]} $\pm$ {stdevs[2]} \\\\
    \hline
    GINE & shared sum & {means[3]} $\pm$ {stdevs[3]} \\\\
    GATv2 & shared sum & {means[4]} $\pm$ {stdevs[4]} \\\\
    Transformer & shared sum & {means[5]} $\pm$ {stdevs[5]} \\\\
    \hline
    Random & - & {means[6]} $\pm$ {stdevs[6]} \\\\
    \hline
    {means} - {stdevs} - {state_switchess}
    \end{{tabular}}
    \end{{table}}"""

        with open(f"/Users/sega/Code/si_bees/reports/report_transmission_rl_{i}.tex", "w") as file:
            file.write(table)

def evaluate_transmission_MARL(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_transmission_marl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    
    # 500k step trainingh with ASHA, 300 Trials
    checkpoints_marl = [
        os.path.join("checkpoints", "20240226_transmission_marl_1_16-38-54-r130", "checkpoint_000009"), # MARL binary
        os.path.join("checkpoints", "20240226_transmission_marl_2_16-38-59-r79", "checkpoint_000009"),  # MARL sum
        os.path.join("checkpoints", "20240226_transmission_marl_3_16-39-00-r62", "checkpoint_000009"),  # MARL individual
    ]

    # run all curriculum levels
    for i, selected_config in enumerate(task_configs):
        if exclude and i in exclude: continue
        print(f"\n\n\n########### run level {i} ##############\n\n\n")

        tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=i))
        task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
        
        means = list()
        stdevs = list()
        state_switchess = list()

        for checkpoint in checkpoints_marl:
            policy_net = PPO.from_checkpoint(checkpoint) if checkpoint else None
            print(f"run {checkpoint} level {i}")

            ratios = list()
            state_switches = list()

            # Initialize progress bar for iterations
            with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
                for _ in range(EVAL_EPISODES):
                    model = task(config=selected_config,
                                    use_cuda=False,
                                    inference_mode=True,
                                    policy_net=policy_net)
                    done = False
                    while not done:
                        _, _, _, truncated = model.step()
                        done = truncated["__all__"]
                        ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
                    state_switches.append(model.n_state_switches)
                    
                    # Update progress bar
                    pbar.update(1)

            means.append(round(mean(ratios), 2))
            stdevs.append(round(stdev(ratios), 2))
            state_switchess.append(round(mean(state_switches)))
            print(means, stdevs, state_switchess)

        print(means, stdevs, state_switchess)

        with open(f"/Users/sega/Code/si_bees/reports/report_transmission_marl_{i}.tex", "w") as file:
            file.write(f"{means} $\pm$ {stdevs}, {state_switches}")

def evaluate_transmission_extended_MARL(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_transmission_ext_marl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    
    # 250 step trainingh with ASHA, 250 Trials
    checkpoints_marl = [
        os.path.join("checkpoints", "20240229_transmission_ext_marl_1_11-34-07-r143", "checkpoint_000004"), # MARL binary
    ]

    # run all curriculum levels
    for i, selected_config in enumerate(task_configs):
        if exclude and i in exclude: continue
        print(f"\n\n\n########### run level {i} ##############\n\n\n")

        tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=i))
        task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
        
        means = list()
        stdevs = list()
        state_switchess = list()

        for checkpoint in checkpoints_marl:
            policy_net = PPO.from_checkpoint(checkpoint)
            print(f"run {checkpoint} level {i}")

            ratios = list()
            state_switches = list()

            # Initialize progress bar for iterations
            with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
                for _ in range(EVAL_EPISODES):
                    model = task(config=selected_config,
                                    use_cuda=False,
                                    inference_mode=True,
                                    policy_net=policy_net)
                    done = False
                    while not done:
                        _, _, _, truncated = model.step()
                        done = truncated["__all__"]
                        ratios.append((model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0)
                    state_switches.append(model.n_state_switches)
                    
                    # Update progress bar
                    pbar.update(1)

            means.append(round(mean(ratios), 2))
            stdevs.append(round(stdev(ratios), 2))
            state_switchess.append(round(mean(state_switches)))
            print(means, stdevs, state_switchess)

        print(means, stdevs, state_switchess)

        with open(f"/Users/sega/Code/si_bees/reports/report_transmission_ext_marl_{i}.tex", "w") as file:
            file.write(f"{means} $\pm$ {stdevs}, {state_switches}")
