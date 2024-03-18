import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from tqdm import tqdm
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config

EVAL_EPISODES = 100

def evaluate_moving_MARL(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_moving_marl.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    
    # 500k step trainingh with ASHA, 250-300 Trials
    checkpoints_marl = [
        #None,
        #os.path.join("checkpoints", "20240227_moving_marl_1_19-55-59-r76", "checkpoint_000009"),    # spread
        #os.path.join("checkpoints", "20240229_moving_marl_2_16-44-52-r1", "checkpoint_000009"),     # spread-connected
        os.path.join("checkpoints", "20240228_moving_marl_3_23-29-34-r12", "checkpoint_000009"),    # neighbours
        #os.path.join("checkpoints", "20240302_moving_marl_4_16-45-38-r146", "checkpoint_000009"),   # neighbours connected
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
            files = ["num_connect", "connect_correct", "avg_dist", "avg_dist_connect", "avg_dist_correct"]
            with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
                for k in range(EVAL_EPISODES):
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
                        with open(f"/Users/sega/Code/si_bees/reports/report_moving_task–{i}_{f}_{str(checkpoint).replace('/', '_') if checkpoint else 'random'}.csv", "a+") as file:
                            file.write(', '.join([str(k)] + [str(d) for d in data[j]])) 
                            file.write("\n")  
                    state_switches.append(model.n_state_switches)


                    # Update progress bar
                    pbar.update(1)

            means.append(round(mean(ratios), 2))
            stdevs.append(round(stdev(ratios), 2))
            state_switchess.append(round(mean(state_switches)))

        print(means, stdevs, state_switchess)

        table = f"""
{means} $\pm$ {stdevs}, {state_switches}\\\\
    """

        with open(f"/Users/sega/Code/si_bees/reports/report_moving_{i}_nb.tex", "w") as file:
            file.write(table)
        #with open(f"/Users/sega/Code/si_bees/reports/report_moving_{i}.tex", "w") as file:
        #    file.write(table)


def evaluate_moving_history_MARL(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_moving_history_marl.yaml"))
    #env_config = read_yaml_config(os.path.join("reports", "env_config_moving_history_marl_big.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    
    # 500k step trainingh with ASHA, 250-300 Trials
    checkpoints_marl = [
        #os.path.join("checkpoints", "20240227_moving_history_marl_1_19-18-56-r10", "checkpoint_000003"),    # spread
        #os.path.join("checkpoints", "20240228_moving_history_marl_2_04-33-42-r196", "checkpoint_000004"),   # spread-connected, encoding s
        #os.path.join("checkpoints", "20240301_moving_history_marl_2_14-25-32-r219", "checkpoint_000005"),   # spread-connected, encoding l
        #os.path.join("checkpoints", "20240301_moving_history_marl_5_14-25-21-r231", "checkpoint_000009"),   # spread-connected big, encoding l
        #os.path.join("checkpoints", "20240229_moving_history_marl_3_11-32-26-r97", "checkpoint_000007"),    # neighbours
        #os.path.join("checkpoints", "20240302_moving_history_marl_4_02-12-22-r197", "checkpoint_000007"),   # shared neighbours
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
            heatmap = [[0] * selected_config["model"]["grid_size"] for _ in range(selected_config["model"]["grid_size"])]


            # Initialize progress bar for iterations
            files = ["num_connect", "connect_correct", "avg_dist", "avg_dist_connect", "avg_dist_correct"]
            with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
                for k in range(EVAL_EPISODES):
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

                        # build heatmap
                        for a in model.schedule_workers.agents:
                            x, y = a.pos
                            heatmap[x][y] += 1
                    
                    # save stats
                    for j, f in enumerate(files):
                        with open(f"/Users/sega/Code/si_bees/reports/report_moving_history_task–{i}_{f}_{str(checkpoint).replace('/', '_') if checkpoint else 'random'}.csv", "a+") as file:
                            file.write(', '.join([str(k)] + [str(d) for d in data[j]]))  
                            file.write("\n")   
                    state_switches.append(model.n_state_switches)

                    # Update progress bar
                    pbar.update(1)
            
            print(heatmap)

            means.append(round(mean(ratios), 2))
            stdevs.append(round(stdev(ratios), 2))
            state_switchess.append(round(mean(state_switches)))

        print(means, stdevs, state_switchess)

        table = f"""
{means} $\pm$ {stdevs}, {state_switches}\\\\
"""

        with open(f"/Users/sega/Code/si_bees/reports/report_moving_history_{i}_l.tex", "w") as file:
            file.write(table)
        # with open(f"/Users/sega/Code/si_bees/reports/report_moving_history_{i}.tex", "w") as file:
        #     file.write(table)
