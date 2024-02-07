import os
import platform
import shutil
import time

from model import MODEL_TYPE_SIMPLE, MODEL_TYPE_MOVING, MOVING_MODELS, SIMPLE_MODELS

if platform.system() == "Darwin":
    pass
else:
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.sched_setaffinity(0, range(os.cpu_count())) 
    print(f"-> cpu count: ", os.cpu_count())
    print(f"-> cpu affinity: ", os.sched_getaffinity(0))

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
    parser.add_argument('--num_ray_threads',    default=36, help='default processes for ray to use')
    parser.add_argument('--num_cpu_for_local',  default=2, help='num cpus for local worker')
    parser.add_argument('--enable_gpu',         action='store_true', help='enable use of gpu')
    parser.add_argument('--env_config',         default=None, help="path to env config")
    parser.add_argument('--actor_config',       default=None, help="path to actor config")
    parser.add_argument('--critic_config',      default=None, help="path to critic config")
    parser.add_argument('--model_type',         default=0, help="choose which model to train, see model.py for mapping")
    parser.add_argument('--restore',            default=None, help="restore experiment for tune")
    args = parser.parse_args()

    print("-> start tune with following parameters")
    print(args)
    use_cuda = args.enable_gpu and torch.cuda.is_available()
    storage_dir = "/Users/sega/Code/si_bees/log" if args.local else "/itet-stor/kpius/net_scratch/si_bees/log"
    ray_dir = os.path.join(os.path.expanduser('~'), "ray_results")

    if args.local:
        print(f"-> using local")
        #ray.init()
        ray.init(num_cpus=1, local_mode=True)
    elif use_cuda:
        # @todo: investigate gpu utilisation
        print(f"-> using {int(args.num_ray_threads)} cpus and a gpu ({os.environ['CUDA_VISIBLE_DEVICES']})")
        ray.init(num_cpus=int(args.num_ray_threads), num_gpus=1)
    else:
        print(f"-> using {int(args.num_ray_threads)} cpus")
        ray.init(num_cpus=int(args.num_ray_threads))

    # select correct model
    model_type = MODEL_TYPE_SIMPLE if args.env_config in SIMPLE_MODELS \
            else MODEL_TYPE_MOVING if args.env_config in MOVING_MODELS \
            else None

    tune.register_env("Simple_env", lambda env_config: Simple_env(env_config, model_type=model_type))
    
    run_name = f"simple-env-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    env_config = read_yaml_config(os.path.join("src_v2", "configs", args.env_config))
    actor_config = read_yaml_config(os.path.join("src_v2", "configs", args.actor_config))
    critic_config = read_yaml_config(os.path.join("src_v2", "configs",args.critic_config))

    pyg_config = dict()
    pyg_config["actor_config"] = filter_tunables(create_tunable_config(actor_config))
    pyg_config["critic_config"] = create_tunable_config(critic_config)
    pyg_config["encoding_size"] = tune.choice([8, 16, 32])
    pyg_config["recurrent"] = tune.choice([1, 0])
    pyg_config["use_cuda"] = use_cuda
    pyg_config["info"] = {
        "env_config": args.env_config,
        "actor_config": args.actor_config,
        "critic_config": args.critic_config
    }
    model = {"custom_model": GNN_PyG,
             "custom_model_config": pyg_config}
    
    # ppo config
    ppo_config = PPOConfig()
    ppo_config.environment(
            env="Simple_env",
            env_config=env_config,
            disable_env_checking=True,
            env_task_fn=curriculum_oracle_switch)
    # default values: https://github.com/ray-project/ray/blob/e6ae08f41674d2ac1423f3c2a4f8d8bd3500379a/rllib/agents/ppo/ppo.py
    ppo_config.training(
            model=model,
            train_batch_size=500,
            shuffle_sequences=False,
            lr=tune.uniform(0.00003, 0.003),
            gamma=0.99,
            use_critic=True,
            use_gae=True,
            lambda_=tune.uniform(0.9, 1),
            kl_coeff=tune.choice([0.0, 0.2]),
            kl_target=tune.uniform(0.003, 0.03),
            vf_loss_coeff=tune.uniform(0.5, 1),
            clip_param=0.2,
            entropy_coeff=tune.choice([0.0, 0.01]),
            grad_clip=1,
            grad_clip_by="value",
            _enable_learner_api=False)
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.callbacks(SimpleCallback)
    #ppo_config.reporting(keep_per_episode_custom_metrics=True)

    # @todo: investigate gpu utilisation
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
                num_cpus_for_local_worker=int(args.num_cpu_for_local),
                placement_strategy="PACK")

    # run and checkpoint config
    run_config = air.RunConfig(
        name=run_name,
        storage_path=storage_dir,
        local_dir=storage_dir,
        stop=CombinedStopper(
            MaxTimestepsStopper(max_timesteps=2500000),
        ),        
        checkpoint_config=CheckpointConfig(
            #checkpoint_score_attribute="custom_metrics/reward_score_mean",
            #num_to_keep=1,
            checkpoint_frequency=100,   # 500 ts per iteration, e.g. every 50'000 ts
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
                metric='custom_metrics/reward_score_mean',
                mode='max',
                grace_period=35000,
                max_t=2500000,
                reduction_factor=2)
        )

    # restore experiments
    if args.restore:
        print(f"-> restoring experiment {args.restore}")
        print(f"  tuner.pkl in {ray_dir}: {os.path.exists(os.path.join(ray_dir, args.restore, 'tuner.pkl'))}")
        print(f"  tuner.pkl in {storage_dir}: {os.path.exists(os.path.join(storage_dir, args.restore, 'tuner.pkl'))}")
        if os.path.exists(os.path.join(ray_dir, args.restore, "tuner.pkl")):
            shutil.copy(os.path.join(ray_dir, args.restore, "tuner.pkl"), os.path.join(storage_dir, args.restore, "tuner.pkl"))
            print("-> copied tuner.pkl from ~/ray")
            time.sleep(30)

        if os.path.exists(os.path.join(storage_dir, args.restore, "tuner.pkl")):
            print(f"-> restore {args.restore}")
            tuner = tune.Tuner.restore(
                os.path.join(storage_dir, args.restore),
                "PPO",
                resume_unfinished=True,
                resume_errored=True,
                restart_errored=True,
                param_space=ppo_config.to_dict()
            )
        else:
            print(f"-> could not restore {args.restore}, no tuner.pkl file found")
    else:
        tuner = tune.Tuner(
            "PPO",
            run_config=run_config,
            tune_config=tune_config,
            param_space=ppo_config.to_dict()
        )

    tuner.fit()

