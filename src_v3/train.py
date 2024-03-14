import os
import platform


if platform.system() == "Darwin":
    pass
else:
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.sched_setaffinity(0, range(os.cpu_count())) 
    print(f"-> cpu count: ", os.cpu_count())
    print(f"-> cpu affinity: ", os.sched_getaffinity(0))

import shutil
import time
import argparse
from datetime import datetime
import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.rllib.utils.from_config import NotProvided
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.stopper import CombinedStopper
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig

from environment import MARL_ENV, RL_ENV, base_env, load_task_model, marl_env
from gnn import gnn_torch_module
from callback import mpe_callback, score_callback
from task_models import mpe_spread_marl_model, mpe_spread_reduced
from utils import create_tunable_config, read_yaml_config
from stopper import max_timesteps_stopper
from curriculum import curriculum_50_min_percentile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to setup hyperparameter tuning')
    parser.add_argument('--local',              action='store_true', help='run in local mode for debugging purposes (default: False)')
    parser.add_argument('--project_dir',        default=None, help='directory of the repository to store runs to')
    parser.add_argument('--env_config',         default=None, help="path to env config")
    parser.add_argument('--actor_config',       default=None, help="path to actor config")
    parser.add_argument('--critic_config',      default=None, help="path to critic config")
    parser.add_argument('--encoding_config',    default=None, help="path to encoding config")
    parser.add_argument('--num_trials',         default=250, help='number of different ray tune trials')
    parser.add_argument('--max_timesteps',      default=200000, help='max timesteps per trial')
    parser.add_argument('--grace_period',       default=35000, help='min timesteps per trial')
    parser.add_argument('--num_ray_threads',    default=36, help='default processes for ray to use')
    parser.add_argument('--num_cpu_for_local',  default=2, help='num cpus for local worker')
    parser.add_argument('--num_rollouts',       default=0, help='num rollout workers')
    parser.add_argument('--restore',            default=None, help="restore experiment for tune")
    args = parser.parse_args()

    print("-> start tune")
    print("-> user parameters: ", args)

    # initialise ray
    print(f"-> ressources:")
    use_cuda = False # @todo
    if args.local:
        print("    local_mode=True\n    num_cpus=4\n    num_gpus=0")
        ray.init(num_cpus=4, local_mode=True)
    elif use_cuda:
        print(f"    local_mode=False\n    num_cpus={int(args.num_ray_threads)}\n    num_gpus=1")
        ray.init(num_cpus=int(args.num_ray_threads), num_gpus=1)
    else:
        print(f"    local_mode=True\n    num_cpus={int(args.num_ray_threads)}\n    num_gpus=0")
        ray.init(num_cpus=int(args.num_ray_threads))

    # set base directories
    print("-> set storage directory")
    base_dir = os.getcwd() if args.local else args.project_dir
    storage_dir = os.path.join(base_dir, "log")
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    ray_dir = os.path.join(os.path.expanduser('~'), "ray_results")
    print(f"    storage: ", storage_dir)
    print(f"    ray:     ", ray_dir)
    print()

    # read in configs as dicts
    env_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.env_config))
    actor_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.actor_config))
    critic_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.critic_config))
    encoding_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.encoding_config))

    # register environments
    assert env_config["env_type"] in {"rl", "marl"}, f"env_type {env_config['env_type']} is not supported"
    env_type = RL_ENV if env_config["env_type"] == "rl" else MARL_ENV
    tune.register_env("base_env", lambda env_config: base_env(config=env_config))
    tune.register_env("marl_env", lambda env_config: marl_env(config=env_config))
    is_mpe = load_task_model(name=env_config["task_model"], env_type=env_type) == mpe_spread_marl_model or mpe_spread_reduced
    
    # gnn module
    gnn = {
        "custom_model": gnn_torch_module,
        "custom_model_config": {
            "env_type": env_type,
            "actor_config": create_tunable_config(actor_config),
            "critic_config": create_tunable_config(critic_config),
            "encoding_config": create_tunable_config(encoding_config),
            "recurrent_actor": tune.choice([0, 1]),
            "recurrent_critic": tune.choice([0, 1]),
            "use_cuda": use_cuda,
            # little hack to upload the config names to wandb, helping with tractability
            "info": {
                "env_config": args.env_config,
                "actor_config": args.actor_config,
                "critic_config": args.critic_config,
                "encoding_config": args.encoding_config
            }
        }
    }
    
    # ppo config
    ppo_config = PPOConfig()
    ppo_config.environment(
            env=env_type,
            env_config=env_config,
            disable_env_checking=True,
            # @todo: make curriculum adjustable
            env_task_fn=curriculum_50_min_percentile if not is_mpe else NotProvided)
    # default values: https://github.com/ray-project/ray/blob/e6ae08f41674d2ac1423f3c2a4f8d8bd3500379a/rllib/agents/ppo/ppo.py
    train_batch_size = 500 if not is_mpe else 7500
    ppo_config.training(
            model=gnn,
            train_batch_size=train_batch_size,
            shuffle_sequences=False,
            lr=tune.uniform(0.00003, 0.003),
            gamma=0.99,
            use_critic=True,
            use_gae=True,
            lambda_=tune.uniform(0.9, 1),
            kl_coeff=tune.choice([0.0, 0.2]),
            kl_target=tune.uniform(0.003, 0.03),
            vf_loss_coeff=tune.uniform(0.5, 1),
            clip_param=tune.choice([0.1, 0.2]),
            entropy_coeff=tune.choice([0.0, 0.01, 0.1]),
            grad_clip=1,
            grad_clip_by="value",
            _enable_learner_api=False)
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.callbacks(score_callback if not is_mpe else mpe_callback)
    ppo_config.multi_agent(count_steps_by="agent_steps")
    #ppo_config.reporting(keep_per_episode_custom_metrics=True)

    # @todo: investigate gpu utilisation
    if use_cuda:
        ppo_config.rollouts(
            num_rollout_workers=int(args.num_rollouts),
            batch_mode="complete_episodes")
        ppo_config.resources(
                num_gpus=1,
                placement_strategy="PACK")
    else:
        ppo_config.rollouts(
            num_rollout_workers=int(args.num_rollouts),
            batch_mode="complete_episodes")
        ppo_config.resources(
                num_cpus_for_local_worker=int(args.num_cpu_for_local),
                placement_strategy="PACK")

    # run and checkpoint config
    run_name = f"{datetime.now().strftime('%Y%m%d')}_{'_'.join(str(args.env_config).split('.')[0].split('_')[2:])}_{datetime.now().strftime('%H-%M-%S')}"
    run_config = air.RunConfig(
        name=run_name,
        storage_path=storage_dir,
        local_dir=storage_dir,
        stop=CombinedStopper(
            max_timesteps_stopper(max_timesteps=int(args.max_timesteps)),
        ),        
        checkpoint_config=CheckpointConfig(
            num_to_keep=10 if is_mpe else None,
            checkpoint_score_attribute="episode_reward_mean" if is_mpe else None,
            checkpoint_frequency=100 if not is_mpe else 1,
            checkpoint_at_end=True),
        callbacks=[WandbLoggerCallback(
                            project="marl_si_v3",
                            group=run_name,
                            api_key_file=".wandb_key",
                            log_config=True)] if not args.local else []
    )

    # tune config
    tune_config = tune.TuneConfig(
            num_samples=int(args.num_trials),
            scheduler= ASHAScheduler(
                time_attr='timesteps_total',
                metric='custom_metrics/reward_score_mean' if not is_mpe else 'episode_reward_mean',
                mode='max',
                grace_period=int(args.grace_period),
                max_t=int(args.max_timesteps),
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

