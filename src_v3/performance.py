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
from datetime import datetime
import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.stopper import CombinedStopper
from ray.rllib.algorithms.ppo import PPOConfig

from task_models import mpe_spread_marl_model
from environment import MARL_ENV, RL_ENV, base_env, load_task_model, marl_env
from gnn import gnn_torch_module
from callback import mpe_callback, score_callback
from utils import create_tunable_config, read_yaml_config
from stopper import max_timesteps_stopper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to setup hyperparameter tuning')
    parser.add_argument('--local',              action='store_true', help='run in local mode for debugging purposes (default: False)')
    parser.add_argument('--project_dir',        default=None, help='directory of the repository to store runs to')
    parser.add_argument('--env_config',         default=None, help="path to env config")
    parser.add_argument('--actor_config',       default=None, help="path to actor config")
    parser.add_argument('--critic_config',      default=None, help="path to critic config")
    parser.add_argument('--encoding_config',    default=None, help="path to encoding config")
    parser.add_argument('--num_ray_threads',    default=36, help='default processes for ray to use')
    parser.add_argument('--num_cpu_for_local',  default=2, help='num cpus for local worker')
    parser.add_argument('--num_rollouts',       default=0, help='num rollout workers')
    args = parser.parse_args()

    print("-> start performance test")
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

    # fixed setup
    env_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.env_config))
    actor_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.actor_config))
    critic_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.critic_config))
    encoding_config = read_yaml_config(os.path.join(base_dir, "src_v3", "configs", args.encoding_config))
    
    assert env_config["env_type"] in {"rl", "marl"}, f"env_type {env_config['env_type']} is not supported"
    env_type = RL_ENV if env_config["env_type"] == "rl" else MARL_ENV
    tune.register_env("base_env", lambda env_config: base_env(config=env_config))
    tune.register_env("marl_env", lambda env_config: marl_env(config=env_config))
    is_mpe = load_task_model(name=env_config["task_model"], env_type=env_type) == mpe_spread_marl_model
    
    # gnn module
    gnn = {
        "custom_model": gnn_torch_module,
        "custom_model_config": {
            "env_type": env_type,
            "actor_config": create_tunable_config(actor_config),
            "critic_config": create_tunable_config(critic_config),
            "encoding_config": create_tunable_config(encoding_config),
            "recurrent_actor": 0,
            "recurrent_critic": 0,
            "use_cuda": use_cuda,
        }
    }
    
    # ppo config
    ppo_config = PPOConfig()
    ppo_config.environment(
            env=env_type,
            env_config=env_config,
            disable_env_checking=True,)
    ppo_config.training(
            model=gnn,
            train_batch_size=500 if not is_mpe else 7500,
            shuffle_sequences=False,
            lr=0.0001,
            gamma=0.99,
            use_critic=True,
            use_gae=True,
            lambda_=0.99,
            kl_coeff=0.1,
            kl_target=0.03,
            vf_loss_coeff=0.7,
            clip_param=0.1,
            entropy_coeff=0.01,
            grad_clip=1,
            grad_clip_by="value",
            _enable_learner_api=False)
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.callbacks(score_callback if not is_mpe else mpe_callback)
    ppo_config.multi_agent(count_steps_by="agent_steps")

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

    run_name = f"{datetime.now().strftime('%Y%m%d')}_performance_{datetime.now().strftime('%H-%M-%S')}"
    tune_config = tune.TuneConfig(num_samples=20)
    run_config = air.RunConfig(
        name=run_name,
        storage_path=storage_dir,
        local_dir=storage_dir,
        stop=CombinedStopper(
            max_timesteps_stopper(max_timesteps=7500),
        ),        
        callbacks=[WandbLoggerCallback(
                            project="marl_si_v3",
                            group=run_name,
                            api_key_file=".wandb_key",
                            log_config=True)] if not args.local else []
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=ppo_config.to_dict()
    )

    tuner.fit()

