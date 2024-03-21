
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

from xxx_mpe import evaluate_mpe, evaluate_mpe_reduced
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
        checkpoints = [
            ["random", None],
            ["15-00-20-r91", os.path.join("checkpoints", "20240229_lever_pulling_15-00-20-r91", "checkpoint_000004")],    # trans 2
            ["14-57-10-r7", os.path.join("checkpoints", "20240226_lever_pulling_14-57-10-r7", "checkpoint_000004")],      # trans 1
            ["15-07-38-r67", os.path.join("checkpoints", "20240226_lever_pulling_15-07-38-r67", "checkpoint_000004")],    # gatv2
            ["14-57-29-r73", os.path.join("checkpoints", "20240226_lever_pulling_14-57-29-r73", "checkpoint_000004")],    # gine
        ]
        env_config_path = os.path.join("reports", "env_config_lever_pulling.yaml")
        evaluate_lever_pulling(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)

    """Transmission RL"""
    if args.task_name == "transmission_rl":
        checkpoints = [
            ["random", None],
            # shared binary
            ["3_19-01-32-r162", os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-32-r162")],   # GINE
            ["3_19-01-13-r64", os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-13-r64")],     # GATv2
            ["3_19-01-46-r222", os.path.join("checkpoints", "20240226_transmission_rl_3_19-01-46-r222")],   # Transformer
            # shared sum
            ["2_14-58-21-r19", os.path.join("checkpoints", "20240228_transmission_rl_2_14-58-21-r19")],     # GINE
            ["2_19-00-41-r169", os.path.join("checkpoints", "20240226_transmission_rl_2_19-00-41-r169", "checkpoint_000000")],  # GATv2
            ["2_19-01-05-r268", os.path.join("checkpoints", "20240226_transmission_rl_2_19-01-05-r268")],   # Transformer
        ]
        env_config_path = os.path.join("reports", "env_config_transmission_rl.yaml")
        evaluate_transmission_RL(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)

    """Transmission MARL"""
    if args.task_name == "transmission_marl":
        checkpoints = [
            ["random", None],
            ["16-38-54-r130", os.path.join("checkpoints", "20240226_transmission_marl_1_16-38-54-r130", "checkpoint_000009")],      # MARL binary
            ["16-38-59-r79", os.path.join("checkpoints", "20240226_transmission_marl_2_16-38-59-r79", "checkpoint_000009")],        # MARL sum
            ["39-00-r62", os.path.join("checkpoints", "20240226_transmission_marl_3_16-39-00-r62", "checkpoint_000009")],           # MARL individual
        ]
        env_config_path = os.path.join("reports", "env_config_transmission_marl.yaml")
        evaluate_transmission_MARL(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)
        
    if args.task_name == "transmission_ext_marl":
        checkpoints = [
            ["random", None],
            ["11-34-07-r143", os.path.join("checkpoints", "20240229_transmission_ext_marl_1_11-34-07-r143", "checkpoint_000004")], # 
        ]   
        env_config_path = os.path.join("reports", "env_config_transmission_ext_marl.yaml")
        evaluate_transmission_extended_MARL(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)

    """Moving MARL"""
    if args.task_name == "moving":
        checkpoints = [
            ["random", None],
            ["19-55-59-r76", os.path.join("checkpoints", "20240227_moving_marl_1_19-55-59-r76", "checkpoint_000009")],      # spread
            ["16-44-52-r1", os.path.join("checkpoints", "20240229_moving_marl_2_16-44-52-r1", "checkpoint_000009")],        # spread-connected
            ["3_23-29-34-r12", os.path.join("checkpoints", "20240228_moving_marl_3_23-29-34-r12", "checkpoint_000009")],    # neighbours
            ["4_16-45-38-r146", os.path.join("checkpoints", "20240302_moving_marl_4_16-45-38-r146", "checkpoint_000009")],  # neighbours connected
        ]
        evaluate_moving_MARL(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)


    if args.task_name == "moving_history":
        checkpoints = [
            ["19-18-56-r10", os.path.join("checkpoints", "20240227_moving_history_marl_1_19-18-56-r10", "checkpoint_000003")],      # spread
            ["04-33-42-r196", os.path.join("checkpoints", "20240228_moving_history_marl_2_04-33-42-r196", "checkpoint_000004")],    # spread-connected, encoding s
            ["14-25-32-r219", os.path.join("checkpoints", "20240301_moving_history_marl_2_14-25-32-r219", "checkpoint_000005")],    # spread-connected, encoding l
            ["14-25-21-r231", os.path.join("checkpoints", "20240301_moving_history_marl_5_14-25-21-r231", "checkpoint_000009")],    # spread-connected big, encoding l
            ["11-32-26-r97", os.path.join("checkpoints", "20240229_moving_history_marl_3_11-32-26-r97", "checkpoint_000007")],      # neighbours
            ["02-12-22-r197", os.path.join("checkpoints", "20240302_moving_history_marl_4_02-12-22-r197", "checkpoint_000007")],    # shared neighbours
        ]
        evaluate_moving_history_MARL(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)
    
    """MPE Spread"""
    if args.task_name == "mpe_spread":
        checkpoints = [
                ["random", None],
                ["10-00-31-r97-c89", os.path.join("checkpoints", "20240313_mpe_spread_2_10-00-31-r97", "checkpoint_000089")],
            ]
        env_config_path = os.path.join("reports", "env_config_mpe_spread.yaml")
        evaluate_mpe(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)
            
    """MPE Spread Reduced"""
    if args.task_name == "mpe_spread_reduced":
        checkpoints = [
                ["09-44-17-r42-c112", os.path.join("checkpoints", "20240314_mpe_spread_reduced_09-44-17-r52", "checkpoint_000112")],
                ["09-41-42-r229-c42", os.path.join("checkpoints", "20240314_mpe_spread_reduced_09-41-42-r229", "checkpoint_000042")],
            ]
        env_config_path = os.path.join("reports", "env_config_mpe_spread_reduced.yaml")
        evaluate_mpe_reduced(name=checkpoints[task_checkpoint][0],
                    checkpoint_path=checkpoints[task_checkpoint][1],
                    env_config_path=env_config_path,
                    task_level=task_level,
                    eval_episodes=eval_episodes)

