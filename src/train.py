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


def run(logging_config: str,
        actor_config: str,
        critic_config: str,
        encoders_config: str,
        env_config: str,
        tune_samples: int = 1000, 
        min_episodes: int = 100, max_episodes: int = 200, batch_size_episodes: int = 4,
        performance_study: bool = False, ray_threads = None,
        rollout_workers: int = 0, cpus_per_worker: int = 1, cpus_for_local_worker: int = 1):
    """starts a run with the given configurations"""

    if ray_threads:
        ray.init(num_cpus=ray_threads)
    else:
        ray.init()
    
    group_name = f"a{actor_config['model']}_c{critic_config['model']}_e{encoders_config['node_encoder']}_{encoders_config['edge_encoder']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if performance_study:
        group_name = f"perf_batchsize-{batch_size_episodes}_rollouts-{rollout_workers}_cpus-{cpus_per_worker}_cpus_local-{cpus_for_local_worker}"
    run_name = group_name
    storage_path = os.path.join(logging_config["storage_path"])

    tune.register_env("CommunicationV1_env", lambda env_config: CommunicationV1_env(env_config))
    model = {"custom_model": GNN_PyG,
            "custom_model_config": build_model_config(actor_config, critic_config, encoders_config, performance_study)}
    curriculum = curriculum_fn if env_config["curriculum_learning"] and not performance_study else NotProvided
    episode_len = env_config["max_steps"]
    min_timesteps = min_episodes * (episode_len + 1)
    max_timesteps = max_episodes * (episode_len + 1) if not performance_study else min_timesteps + 1
    batch_size = batch_size_episodes * (episode_len + 1)

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
            train_batch_size=batch_size,
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
        storage_path=storage_path,
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
                max_t=max_timesteps,
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
    parser.add_argument('--actor_config', default="model_GINE.json", help="path to the actor model config")
    parser.add_argument('--critic_config', default="model_GAT.json", help="path to the critic model config")
    parser.add_argument('--encoders_config', default="encoders_sincos.json", help="path to the encoders config")
    parser.add_argument('--env_config', default="env_comv1_1.json", help="path to env config")
    parser.add_argument('--performance_study', default=False, action='store_true', help='run performance study with fixed set of parameters and run length')
    parser.add_argument('--ray_threads', default=None, type=int, help="number of threads to use for ray")
    parser.add_argument('--rollout_workers', default=0, type=int, help="number of rollout workers")
    parser.add_argument('--cpus_per_worker', default=1, type=int, help="number of cpus per rollout worker")
    parser.add_argument('--cpus_for_local_worker', default=1, type=int, help="number of cpus for local worker")
    parser.add_argument('--batch_size_episodes', default=4, type=int, help="batch size episodes for training")
    parser.add_argument('--min_episodes', default=100, type=int, help="min number of min_episodes to run")
    parser.add_argument('--max_episodes', default=1000, type=int, help="max number of min_episodes to run")
    parser.add_argument('--tune_samples', default=1000, type=int, help="number of samples to run")
    

    args = parser.parse_args()

    # load configs
    config_dir = os.path.join("src", "configs")
    actor_config = load_config_dict(os.path.join(config_dir, args.actor_config))
    critic_config = load_config_dict(os.path.join(config_dir, args.critic_config))
    encoders_config = load_config_dict(os.path.join(config_dir, args.encoders_config))
    env_config = load_config_dict(os.path.join(config_dir, args.env_config))
    
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
        performance_study=args.performance_study,
        tune_samples=args.tune_samples,
        min_episodes=args.min_episodes,
        max_episodes=args.max_episodes,
        batch_size_episodes=args.batch_size_episodes,
        ray_threads=args.ray_threads, 
        rollout_workers=args.rollout_workers, 
        cpus_per_worker=args.cpus_per_worker,
        cpus_for_local_worker=args.cpus_for_local_worker)



