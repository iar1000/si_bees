from statistics import mean, stdev
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from environment import MARL_ENV, load_task_model, marl_env
from utils import read_yaml_config

def evaluate_mpe(name: str, checkpoint_path: str, env_config_path: str, 
                 task_level: int = 0, eval_episodes: int = 50):
    print(f"\n\n\n-> run eval:")
    print(f"\t- checkpoint: ", checkpoint_path)
    print(f"\t- env config: ", env_config_path)
    print(f"\t- task level: ", task_level)
    print(f"\t- eval episodes: ", eval_episodes)
    
    env_config = read_yaml_config(env_config_path)
    task_configs = [env_config["task_configs"][t] for t in env_config["task_configs"]]
    selected_config = task_configs[task_level]

    tune.register_env("marl_env", lambda _: marl_env(config=env_config, initial_task_level=task_level))
    task = load_task_model(name=env_config["task_model"], env_type=MARL_ENV)
    policy_net = PPO.from_checkpoint(checkpoint_path) if checkpoint_path else None
        
    rewards = list()
    for k in range(eval_episodes):
        print(f"episode {k}/{eval_episodes}")
        model = task(config=selected_config,
                        use_cuda=False,
                        inference_mode=True,
                        policy_net=policy_net)
        done = False
        while not done:
            _, rewardss, _, truncated = model.step()
            done = truncated["__all__"]
            rewards.append(-mean(rewardss.values()))

    means = round(mean(rewards), 2)
    stdevs = round(stdev(rewards), 2)
    print(means, stdevs)

    with open(f"reports/report_mpe_{name}_{task_level}.txt", "a+") as file:
        file.write(f"\t- checkpoint: {checkpoint_path}\n")
        file.write(f"\t- env config: {env_config_path}\n")
        file.write(f"\t- task level: {task_level}\n")
        file.write(f"\t- eval episodes: {eval_episodes}\n")
        file.write(f"mean: {means}\n")
        file.write(f"stdev: {stdevs}\n")