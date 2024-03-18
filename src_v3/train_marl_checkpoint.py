import json
import os
import platform

if platform.system() == "Darwin":
    pass
else:
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.sched_setaffinity(0, range(os.cpu_count())) 
    print(f"-> cpu count: ", os.cpu_count())
    print(f"-> cpu affinity: ", os.sched_getaffinity(0))

import argparse
import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.stopper import CombinedStopper
from datetime import datetime
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector

import torch

from callback import SimpleCallback
from pyg_marl import GNN_PyG
from stopper import MaxTimestepsStopper
from curriculum import curriculum_oracle_switch

# surpress excessive logging
#wandb_logger = logging.getLogger("wandb")
#wandb_logger.setLevel(logging.WARNING)
# wandbactor_logger = logging.getLogger("_WandbLoggingActor")
# wandbactor_logger.setLevel(logging.DEBUG)


# script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to setup hyperparameter tuning')
    parser.add_argument('--local',              action='store_true', help='execution location (default: False)')
    parser.add_argument('--cp_path',            default=None, help='path to checkpoint where params.json file is stored')
    parser.add_argument('--num_ray_threads',    default=36, help='default processes for ray to use')
    parser.add_argument('--num_samples',        default=10, help='num samples to run')
    parser.add_argument('--num_workers',         default=4, help='num workers to train with')
    parser.add_argument('--enable_gpu',         action='store_true', help='enable use of gpu')
    args = parser.parse_args()

    storage_dir = "/Users/sega/Code/si_bees/log" if args.local else "/itet-stor/kpius/net_scratch/si_bees/log"
    ray_dir = os.path.join(os.path.expanduser('~'), "ray_results")
    use_cuda = args.enable_gpu and torch.cuda.is_available()

    if use_cuda:
        print(f"-> using 4 cpus and a gpu 1")
        ray.init(num_cpus=4, num_gpus=1)
    else:
        print(f"-> using {int(args.num_ray_threads)} cpus")
        ray.init(num_cpus=int(args.num_ray_threads))

    
    run_name = f"marl-params-{datetime.now().strftime('%Y%m-%d-%H-%M%S')}"

    # load params
    params_path = os.path.join(args.cp_path, "params.json")
    params = {}
    with open(params_path) as f:
        params = json.loads(f.read())

    # restore non-seriasables
    tune.register_env("Marl_env", lambda env_config: Marl_env(config=env_config, env_config_file=params["model"]["custom_model_config"]["info"]["env_config"]))
    params["callbacks"] = SimpleCallback
    params["env_task_fn"] = curriculum_oracle_switch
    params["model"]["custom_model"] = GNN_PyG
    params["model"]["custom_model_config"]["info"]["checkpoint"] = args.cp_path
    params["policy_mapping_fn"] = AlgorithmConfig.DEFAULT_POLICY_MAPPING_FN
    params["sample_collector"] = SimpleListCollector

    # override variables
    params["num_cpus_for_driver"] = 2
    params["num_workers"] = 0
    params["num_gpus"] = 1 if use_cuda else 0
    def rec_update(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                rec_update(value)
            elif key == "n_workers":
                dictionary[key] = int(args.num_workers)
    rec_update(params)


    # run and checkpoint config
    run_config = air.RunConfig(
        name=run_name,
        storage_path=storage_dir,
        local_dir=storage_dir,
        stop=CombinedStopper(
            MaxTimestepsStopper(max_timesteps=2500000),
        ),        
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=100,   # 500 ts per iteration, e.g. every 50'000 ts
            checkpoint_at_end=True),
        callbacks=[WandbLoggerCallback(
                            project="si_marl",
                            group=run_name,
                            api_key_file=".wandb_key",
                            log_config=True)] if not args.local else []
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=run_config,
        tune_config=tune.TuneConfig(num_samples=int(args.num_samples)),
        param_space=params
    )

    tuner.fit()

