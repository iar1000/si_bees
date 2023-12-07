import argparse
import logging
import os
import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.from_config import NotProvided
from datetime import datetime

from configs.utils import load_config_dict
from callbacks import ReportModelStateCallback
from curriculum import curriculum_fn
from envs.communication_v1.environment import CommunicationV1_env
from envs.communication_v1.models.pyg import GNN_PyG
from utils import create_tunable_config, filter_actor_gnn_tunables

# set logging
wandb_logger = logging.getLogger("wandb")
wandb_logger.setLevel(logging.WARNING)

def run(logging_config: dict, 
        actor_config: dict,
        critic_config: dict,
        encoders_config: dict,
        env_config: dict,
        tune_config: dict):
    """starts a run with the given configurations"""

    ray.init()
    
    group_name = f"a-{actor_config['model']}_c-{critic_config['model']}_e-{encoders_config['edge_encoder']}"
    run_name = f"{group_name}_{datetime.now().strftime('%Y%m%d%H%M-%S')}"
    storage_path = os.path.join(logging_config["storage_path"])

    tune.register_env("CommunicationV1_env", lambda env_config: CommunicationV1_env(env_config))
    tunable_model_config = {}
    tunable_model_config["actor_config"] = filter_actor_gnn_tunables(create_tunable_config(actor_config))
    tunable_model_config["critic_config"] = create_tunable_config(critic_config)
    tunable_model_config["encoders_config"] = create_tunable_config(encoders_config)
    model = {"custom_model": GNN_PyG,
            "custom_model_config": tunable_model_config}

    # ppo config
    ppo_config = (
        PPOConfig()
        .environment(
            "CommunicationV1_env", # @todo: need to build wrapper
            env_config=env_config,
            env_task_fn=curriculum_fn if env_config["curriculum_learning"] else NotProvided,
            disable_env_checking=True)
        .training(
            gamma=tune.uniform(0.1, 0.9),
            lr=tune.uniform(1e-4, 1e-1),
            grad_clip=1,
            grad_clip_by="value",
            model=model,
            train_batch_size=450,
            _enable_learner_api=False,
        )
        .rollouts(
            num_rollout_workers=2
        )
        .resources(
            num_cpus_per_worker=1,
            num_cpus_for_local_worker=2,
            placement_strategy="PACK",
        )
        .rl_module(_enable_rl_module_api=False)
        .callbacks(ReportModelStateCallback)
    )

    # logging callback
    callbacks = list()
    if logging_config["enable_wandb"]:
        callbacks.append(WandbLoggerCallback(
                            project=logging_config["project"],
                            group=run_name,
                            api_key_file=logging_config["api_key_file"],
                            log_config=True,
    ))
        
    # run and checkpoint config
    run_config = air.RunConfig(
        name=run_name,
        stop={"timesteps_total": tune_config["max_timesteps"]}, # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        storage_path=storage_path,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute="custom_metrics/curr_learning_score_mean",
            num_to_keep=10,
            checkpoint_frequency=50,
            checkpoint_at_end=True),
    )

    # tune config
    tune_config = tune.TuneConfig(
            num_samples=tune_config["num_samples"],
            scheduler= ASHAScheduler(
                time_attr='timesteps_total',
                metric='custom_metrics/curr_learning_score_mean',
                mode='max',
                max_t=tune_config["max_timesteps"],
                grace_period=tune_config["min_timesteps"],
                reduction_factor=2)
        )

    tuner = tune.Tuner(
        "PPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=ppo_config.to_dict()
    )

    tuner.fit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to setup hyperparameter tuning')
    parser.add_argument('--location', default="local", choices=['cluster', 'local'], help='execution location, setting depending variables')
    parser.add_argument('--actor_config', default=None, help="path to the actor model config")
    parser.add_argument('--critic_config', default=None, help="path to the critic model config")
    parser.add_argument('--encoders_config', default=None, help="path to the encoders config")
    parser.add_argument('--env_config', default=None, help="path to env config")
    parser.add_argument('--tune_config', default="tune_ppo.json", help="path to tune config")

    args = parser.parse_args()

    # load configs
    config_dir = os.path.join("src", "configs")
    actor_config = load_config_dict(os.path.join(config_dir, args.actor_config))
    critic_config = load_config_dict(os.path.join(config_dir, args.critic_config))
    encoders_config = load_config_dict(os.path.join(config_dir, args.encoders_config))
    env_config = load_config_dict(os.path.join(config_dir, args.env_config))
    tune_config = load_config_dict(os.path.join(config_dir, args.tune_config))
    
    # logging config
    if args.location == 'cluster':
        logging_config = load_config_dict(os.path.join(config_dir, "logging_cluster.json"))
    else:
        logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

    # sanity print
    print("===== run hyperparameter tuning =======")
    for k, v in args.__dict__.items():
        print(f"\t{k}: {v}")
    print("\n")

    run(logging_config=logging_config,
        actor_config=actor_config,
        critic_config=critic_config,
        encoders_config=encoders_config,
        env_config=env_config,
        tune_config=tune_config)



