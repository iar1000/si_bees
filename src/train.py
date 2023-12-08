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

# create tunable configs
def build_model_config(actor_config: dict, critic_config: dict, encoders_config: dict, performance_study: bool):
    tunable_model_config = dict()
    # create fixed set of model parameters for performance study
    if performance_study:
        tunable_model_config["actor_config"] = {
            "model": "GINEConv",
            "mlp_hiddens": 2,
            "mlp_hiddens_size": 32}
        tunable_model_config["critic_config"] = {
            "model": "GATConv",
            "critic_rounds": 2,
            "critic_fc": True,
            "dropout": 0.003}
        tunable_model_config["encoders_config"] = {
            "encoding_size": 8,
            "node_encoder": "fc",
            "node_encoder_hiddens": 2,
            "node_encoder_hiddens_size": 16,
            "edge_encoder": "sincos"}
    # make configs tunable
    else:
        tunable_model_config["actor_config"] = filter_actor_gnn_tunables(create_tunable_config(actor_config))
        tunable_model_config["critic_config"] = create_tunable_config(critic_config)
        tunable_model_config["encoders_config"] = create_tunable_config(encoders_config)
    
    return tunable_model_config


def run(logging_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
        encoders_config_path: str,
        env_config_path: str,
        tune_samples: int = 1000, min_episodes: int = 100, max_episodes: int = 200,
        performance_study: bool = False, ray_threads = None,
        rollout_workers: int = 0, cpus_per_worker: int = 1, cpus_for_local_worker: int = 1):
    """starts a run with the given configurations"""

    if ray_threads:
        ray.init(num_cpus=ray_threads)
    else:
        ray.init()
    
    group_name = f"perf_study_rollies-{rollout_workers}_cpus-{cpus_per_worker}_cpus_local-{cpus_for_local_worker}"
    run_name = f"{group_name}_{datetime.now().strftime('%Y%m%d%H%M-%S')}"
    storage_path = os.path.join(logging_config["storage_path"])

    tune.register_env("CommunicationV1_env", lambda env_config: CommunicationV1_env(env_config))
    env_config = load_config_dict(env_config_path)
    logging_config = load_config_dict(logging_config_path)
    model = {"custom_model": GNN_PyG,
            "custom_model_config": build_model_config(
                load_config_dict(actor_config_path), 
                load_config_dict(critic_config_path), 
                load_config_dict(encoders_config_path), 
                performance_study)}
    curriculum = curriculum_fn if env_config["curriculum_learning"] and not performance_study else NotProvided
    episode_len = env_config["max_steps"]
    min_timesteps = min_episodes * episode_len
    max_timesteps = max_episodes * episode_len

    # ppo config
    ppo_config = (
        PPOConfig()
        .environment(
            "CommunicationV1_env",
            env_config=env_config,
            disable_env_checking=True,
            env_task_fn=curriculum
        )
        .training(
            gamma=tune.uniform(0.1, 0.9),
            lr=tune.uniform(1e-4, 1e-1),
            grad_clip=1,
            grad_clip_by="value",
            model=model,
            train_batch_size=tune.choice([2 * episode_len, 10 * episode_len]),
            _enable_learner_api=False,
        )
        .rollouts(num_rollout_workers=rollout_workers)
        .resources(
            num_cpus_per_worker=cpus_per_worker,
            num_cpus_for_local_worker=cpus_for_local_worker,
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
        stop={"timesteps_total": max_timesteps}, # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        #storage_path=storage_path,
        callbacks=callbacks,
        # checkpoint_config=CheckpointConfig(
        #     checkpoint_score_attribute="custom_metrics/curr_learning_score_mean",
        #     num_to_keep=10,
        #     checkpoint_frequency=50,
        #     checkpoint_at_end=True),
    )

    # tune config
    tune_config = tune.TuneConfig(
            num_samples=tune_samples,
            scheduler= ASHAScheduler(
                time_attr='timesteps_total',
                metric='custom_metrics/curr_learning_score_mean',
                mode='max',
                grace_period=min_timesteps,
                max_t=min_timesteps + 1 if performance_study else max_timesteps,
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
    parser.add_argument('--performance_study', default=False, action='store_true', help='run performance study with fixed set of parameters and run length')
    parser.add_argument('--ray_threads', default=None, type=int, help="number of threads to use for ray")
    parser.add_argument('--rollout_workers', default=0, type=int, help="number of rollout workers")
    parser.add_argument('--cpus_per_worker', default=1, type=int, help="number of cpus per rollout worker")
    parser.add_argument('--cpus_for_local_worker', default=1, type=int, help="number of cpus for local worker")
    args = parser.parse_args()

    # sanity print
    print("===== run hyperparameter tuning =======")
    for k, v in args.__dict__.items():
        print(f"\t{k}: {v}")
    print("\n")

    config_dir = os.path.join("src", "configs")
    # logging config
    if args.location == 'cluster':
        logging_config_path = load_config_dict(os.path.join(config_dir, "logging_cluster.json"))
    else:
        logging_config_path = load_config_dict(os.path.join(config_dir, "logging_local.json"))

    run(logging_config_path=logging_config_path,
        actor_config_path=os.path.join(config_dir, args.actor_config),
        critic_config_path=os.path.join(config_dir, args.critic_config),
        encoders_config_path=os.path.join(config_dir, args.encoders_config),
        env_config_path=os.path.join(config_dir, args.env_config),
        performance_study=args.performance_study,
        ray_threads=args.ray_threads, 
        rollout_workers=args.rollout_workers, 
        cpus_per_worker=args.cpus_per_worker,
        cpus_for_local_worker=args.cpus_for_local_worker)



