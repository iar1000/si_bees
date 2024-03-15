
import argparse
import os
import platform
if platform.system() == "Darwin":
    pass
else:
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.sched_setaffinity(0, range(os.cpu_count())) 
    print(f"-> cpu count: ", os.cpu_count())
    print(f"-> cpu affinity: ", os.sched_getaffinity(0))

import ray
ray.init()

from xxx_mpe import evaluate_mpe
from xxx_lever_pulling import evaluate_lever_pulling
from xxx_moving import evaluate_moving_MARL, evaluate_moving_history_MARL
from xxx_transmission import evaluate_transmission_MARL, evaluate_transmission_RL, evaluate_transmission_extended_MARL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to evaluate task models')
    parser.add_argument('--task_name',          type=str, help='Name of the task')
    parser.add_argument('--task_checkpoint',    type=int, help='Checkpoint for the task')
    parser.add_argument('--task_level',         type=int, help='Level of the task')
    parser.add_argument('--eval_episodes',      type=int, help='Number of evaluation episodes')
    args = parser.parse_args()


    task_checkpoint = int(args.task_checkpoint)
    task_level = int(args.task_level)
    eval_episodes = int(args.eval_episodes)
    
    """Lever Pulling Task"""
    if args.task_name == "lever":
        evaluate_lever_pulling()

    """Transmission RL"""
    if args.task_name == "transmission_rl":
        evaluate_transmission_RL(exclude=[0, 1])

    """Transmission MARL"""
    if args.task_name == "transmission_marl":
        evaluate_transmission_MARL(exclude=[0, 1])
    if args.task_name == "transmission_ext_marl":
        evaluate_transmission_extended_MARL(exclude=[0, 1])

    """Moving MARL"""
    if args.task_name == "moving":
        evaluate_moving_MARL(exclude=[0,1,2,4,5])
    if args.task_name == "moving_history":
        evaluate_moving_history_MARL(exclude=[0, 1, 2, 3])
    
    """MPE Spread"""
    if args.task_name == "mpe_spread":
        checkpoints = [
                ["random", None],
                ["10-00-31-r97-c89", os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000089")],
                ["10-00-31-r97-c92", os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000092")],
            ]
        env_config_path = os.path.join("reports", "env_config_mpe_spread.yaml")
        evaluate_mpe(name=checkpoints[task_checkpoint][0],
                      checkpoint_path=checkpoints[task_checkpoint][1],
                      env_config_path=env_config_path,
                      task_level=task_level,
                      eval_episodes=eval_episodes)

