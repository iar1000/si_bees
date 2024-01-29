import os
os.environ["WANDB__SERVICE_WAIT"] = "600"

import argparse
import logging
import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.stopper import CombinedStopper
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from datetime import datetime

import torch

from callback import SimpleCallback
from environment import Simple_env
from pyg import GNN_PyG
from utils import create_tunable_config, filter_tunables, read_yaml_config
from stopper import MaxTimestepsStopper, RewardComboStopper, RewardMinStopper

# surpress excessive logging
#wandb_logger = logging.getLogger("wandb")
#wandb_logger.setLevel(logging.WARNING)
wandbactor_logger = logging.getLogger("_WandbLoggingActor")
wandbactor_logger.setLevel(logging.DEBUG)

# script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to setup hyperparameter tuning')
    parser.add_argument('--local',              action='store_true', help='execution location (default: False)')
    parser.add_argument('--num_ray_threads',    default=36, help='default processes for ray to use')
    parser.add_argument('--enable_gpu',         action='store_true', help='enable use of gpu')
    parser.add_argument('--env_config',         default=None, help="path to env config")
    parser.add_argument('--actor_config',       default=None, help="path to actor config")
    parser.add_argument('--critic_config',      default=None, help="path to critic config")
    args = parser.parse_args()

    print("-> start tune with following parameters")
    print(args)
    use_cuda = args.enable_gpu and torch.cuda.is_available()
    storage_dir = "/Users/sega/Code/si_bees/log" if args.local else "/itet-stor/kpius/net_scratch/si_bees/log"
    #os.environ["WANDB_DIR"] = storage_dir
    #os.environ["WANDB_CACHE_DIR"] = storage_dir
    #os.environ["WANDB_CONFIG_DIR"] = storage_dir

    if args.local:
        print(f"-> using autoscale")
        ray.init()
        #ray.init(num_cpus=1, local_mode=True)
    elif use_cuda:
        print(f"-> using {int(args.num_ray_threads)} cpus and a gpu ({os.environ['CUDA_VISIBLE_DEVICES']})")
        ray.init(num_cpus=int(args.num_ray_threads), num_gpus=1)
    else:
        print(f"-> using {int(args.num_ray_threads)} cpus")
        ray.init(num_cpus=int(args.num_ray_threads))

    tune.register_env("Simple_env", lambda env_config: Simple_env(env_config))
    
    run_name = f"simple-env-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    env_config = read_yaml_config(os.path.join("src_v2", "configs", args.env_config))
    actor_config = read_yaml_config(os.path.join("src_v2", "configs", args.actor_config))
    critic_config = read_yaml_config(os.path.join("src_v2", "configs",args.critic_config))

    pyg_config = dict()
    pyg_config["actor_config"] = filter_tunables(create_tunable_config(actor_config))
    pyg_config["critic_config"] = create_tunable_config(critic_config)
    pyg_config["use_cuda"] = use_cuda
    pyg_config["info"] = ""
    model = {"custom_model": GNN_PyG,
             "custom_model_config": pyg_config}
    
    # ppo config
    ppo_config = PPOConfig()
    ppo_config.environment(
            env="Simple_env",
            env_config=env_config,
            disable_env_checking=True)
    # default values: https://github.com/ray-project/ray/blob/e6ae08f41674d2ac1423f3c2a4f8d8bd3500379a/rllib/agents/ppo/ppo.py
    ppo_config.training(
            model=model,
            train_batch_size=tune.choice([256, 512, 2048]),
            shuffle_sequences=True,
            lr=tune.uniform(5e-6, 0.003),
            gamma=0.99,
            use_critic=True,
            use_gae=True,
            lambda_=tune.uniform(0.9, 1),
            kl_coeff=tune.choice([0.0, 0.2, 0.4]),
            kl_target=tune.uniform(0.003, 0.03),
            vf_loss_coeff=tune.uniform(0.5, 1),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            entropy_coeff=tune.choice([0.0, 0.01]),
            grad_clip=1,
            grad_clip_by="value",
            _enable_learner_api=False)
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.callbacks(SimpleCallback)
    ppo_config.reporting(keep_per_episode_custom_metrics=True)

    # @todo
    if use_cuda:
        ppo_config.rollouts(num_rollout_workers=0)
        ppo_config.resources(
                num_gpus=0.2,
                #num_cpus_for_local_worker=2,
                #num_learner_workers=0,
                #num_gpus_per_learner_worker=1,
                #num_cpus_per_worker=1,
                placement_strategy="PACK")
    else:
        ppo_config.rollouts(num_rollout_workers=0)
        ppo_config.resources(
            num_cpus_for_local_worker=10,
            placement_strategy="PACK")

    # run and checkpoint config
    run_config = air.RunConfig(
        name=run_name,
        storage_path=storage_dir,
        local_dir=storage_dir,
        stop=CombinedStopper(
            MaxTimestepsStopper(max_timesteps=5000000),
        ),        
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute="episode_reward_mean",
            num_to_keep=1,
            checkpoint_frequency=10,
            checkpoint_at_end=True),
        callbacks=[WandbLoggerCallback(
                            project="si_marl",
                            group=run_name,
                            api_key_file=".wandb_key",
                            log_config=True)] if not args.local else []
    )

    # tune config
    tune_config = tune.TuneConfig(
            num_samples=100000,
            scheduler= ASHAScheduler(
                time_attr='timesteps_total',
                metric='episode_reward_mean',
                mode='max',
                grace_period=35000,
                max_t=5000000,
                reduction_factor=2)
        )

    tuner = tune.Tuner(
        "PPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=ppo_config.to_dict()
    )

    tuner.fit()

