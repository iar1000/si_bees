import os
from statistics import stdev, mean
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
#from tqdm import tqdm
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config

EVAL_EPISODES = 50

ray.init(num_cpus=4, local_mode=True)

def evaluate_mpe(exclude = None):
    env_config = read_yaml_config(os.path.join("reports", "env_config_mpe_spread.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]

    checkpoints_marl = [
        None,
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000086"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000088"),
        os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000089"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000090"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000091"),
        os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000092"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000093"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000094"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000095"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000096"),
        #os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000101"),
    ]
    #[0.74, 0.28, 0.26, 0.28, 0.27, 0.26, 0.26, 0.26, 0.27, 0.28, 0.27, 0.31] 
    #[0.26, 0.21, 0.2, 0.21, 0.21,  0.19, 0.2, 0.21, 0.2, 0.21, 0.21, 0.24] [0.27, 0.14, 0.14, 0.1, 0.13, 0.15, 0.11, 0.13, 0.14, 0.12, 0.08, 0.13] [0.12, 0.12, 0.12, 0.09, 0.11, 0.12, 0.1, 0.11, 0.12, 0.1, 0.07, 0.11]

    for i, selected_config in enumerate(task_configs):
        if exclude and i in exclude: continue
        print(f"\n\n\n########### run level {i} ##############\n\n\n")

        tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=i))
        task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)

        means = list()
        stdevs = list()
        means_n = list()
        stdevs_n = list()
        for checkpoint in checkpoints_marl:
            policy_net = PPO.from_checkpoint(checkpoint) if checkpoint else None
            print(f"run {checkpoint} level {i}")
            
            rewards = list()

            # Initialize progress bar for iterations
            #with tqdm(total=EVAL_EPISODES, desc="episodes") as pbar:
            for k in range(EVAL_EPISODES):
                model = task(config=selected_config,
                                use_cuda=False,
                                inference_mode=True,
                                policy_net=policy_net)
                done = False
                while not done:
                    _, rewardss, _, truncated = model.step()
                    done = truncated["__all__"]
                    rewards.append(-mean(rewardss.values()))

                    # Update progress bar
                   # pbar.update(1)

            means.append(round(mean(rewards), 2))
            stdevs.append(round(stdev(rewards), 2))
            
            r_min = min(rewards)
            r_max = max(rewards)
            normalised_rewards = [(r - r_min) / (r_max - r_min) for r in rewards]
            means_n.append(round(mean(normalised_rewards), 2))
            stdevs_n.append(round(stdev(normalised_rewards), 2))

        print(means, stdevs, means_n, stdevs_n)

        table = f"""
{means} $\pm$ {stdevs}\\\\
{means_n} $\pm$ {stdevs_n}\\\\
    """

        #with open(f"/Users/sega/Code/si_bees/reports/report_mpe_{i}.tex", "w") as file:
        with open(f"reports/report_mpe_{i}.tex", "w") as file:
            file.write(table)

evaluate_mpe()