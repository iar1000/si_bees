import os
from statistics import stdev, mean
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config


def evaluate_lever_pulling():
    checkpoint_dir = os.path.join("checkpoints", "20240229_lever_pulling_15-00-20-r91")
    env_config = read_yaml_config(os.path.join(checkpoint_dir, "env_config_lever_pulling.yaml"))
    task_configs = [env_config["task_configs"][task_level] for task_level in env_config["task_configs"]]
    tune.register_env("marl_env", lambda _: marl_env(config=env_config))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    
    checkpoints = [
        os.path.join("checkpoints", "20240229_lever_pulling_15-00-20-r91", "checkpoint_000004"),    # trans 2
        os.path.join("checkpoints", "20240226_lever_pulling_14-57-10-r7", "checkpoint_000004"),     # trans 1
        os.path.join("checkpoints", "20240226_lever_pulling_15-07-38-r67", "checkpoint_000004"),    # gatv2
        os.path.join("checkpoints", "20240226_lever_pulling_14-57-29-r73", "checkpoint_000004"),    # gine
    ]

    means = list()
    stdevs = list()

    for checkpoint in checkpoints:
        policy_net = PPO.from_checkpoint(checkpoint)
        ratios = list()
        for _ in range(1000):
            model = task(config=task_configs[0],
                                use_cuda=False,
                                inference_mode=True,
                                policy_net=policy_net)
            _, rewards, _, _ = model.step()
            ratios.append(rewards[0])

        means.append(round(mean(ratios), 2))
        stdevs.append(round(stdev(ratios), 2))

    table = f"""
\\begin{{table}}[h]
\centering
\\begin{{tabular}}{{|c|c|c|c|}}
\hline
\\textbf{{Model}} & \\textbf{{Convolution}} & \\textbf{{Rounds}} & \\textbf{{Result}} \\\\
\hline
CommNet & - & 2 & 0.94 \\\\
Ours & Transformer & 2 & {means[0]} $\pm$ {stdevs[0]} \\\\
Ours & Transformer & 1 & {means[1]} $\pm$ {stdevs[1]} \\\\
Ours & GATv2 & 1 & {means[2]} $\pm$ {stdevs[2]} \\\\
Ours & GINE & 1 & {means[3]} $\pm$ {stdevs[3]} \\\\
\hline
\end{{tabular}}
\caption{{Lever Pulling Task Results. Shows mean and standard deviation over 1000 independent samples for distinct levers pulled, GNN convolution type ('Convolution'), and communication rounds ('Rounds').}}
\label{{tab:results-lever}}
\end{{table}}"""

    with open(f"/Users/sega/Code/si_bees/reports/report_lever_pulling.tex", "w") as file:
        file.write(table)