import argparse
import logging
import os
import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.tune.stopper import CombinedStopper
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.from_config import NotProvided
from datetime import datetime

from configs.utils import load_config_dict
from callbacks_v2 import ReportModelStateCallback
from curriculum_v2 import curriculum_fn
from envs.communication_v2.environment import CommunicationV2_env
from envs.communication_v2.models.pyg import GNN_PyG
from stopper_v2 import MaxTimestepsStopper, RewardMinStopper
from utils import create_tunable_config, filter_tunables

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
        tunable_model_config["actor_config"] = create_tunable_config(filter_tunables(actor_config))
        tunable_model_config["critic_config"] = create_tunable_config(critic_config)
        tunable_model_config["encoders_config"] = create_tunable_config(encoders_config)
    
    return tunable_model_config


def run(logging_config: str,
        actor_config: str,
        critic_config: str,
        encoders_config: str,
        env_config: str,
        min_timesteps: int, max_timesteps: int, batch_size: int,
        rollout_workers: int, cpus_per_worker: int, cpus_for_local_worker: int,
        tune_samples: int, 
        performance_study: bool = False, ray_threads = None):
    """starts a run with the given configurations"""

    if ray_threads:
        ray.init(num_cpus=ray_threads)
    else:
        ray.init()
    
    group_name = f"a{actor_config['model']}_c{critic_config['model']}_e{encoders_config['node_encoder']}_{encoders_config['edge_encoder']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if performance_study:
        group_name = f"perf_rollouts-{rollout_workers}_cpus-{cpus_per_worker}_cpus_local-{cpus_for_local_worker}"
    run_name = group_name
    storage_path = os.path.join(logging_config["storage_path"])

    tune.register_env("CommunicationV2_env", lambda env_config: CommunicationV2_env(env_config))
    model = {"custom_model": GNN_PyG,
            "custom_model_config": build_model_config(actor_config, critic_config, encoders_config, performance_study)}
    curriculum = curriculum_fn if env_config["curriculum_learning"] and not performance_study else NotProvided
    max_timesteps = min_timesteps + 1 if performance_study else max_timesteps

    # ppo config
    # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    ppo_config = PPOConfig()
    ppo_config.environment(
            "CommunicationV2_env",
            env_config=env_config,
            disable_env_checking=True,
            env_task_fn=curriculum
        )
    ppo_config.training(
            model=model,
            train_batch_size=batch_size,
            shuffle_sequences=True,
            use_critic=True,
            use_gae=True,
            lambda_=tune.uniform(0.9, 1),
            gamma=0.99,
            lr=tune.uniform(5e-6, 0.003),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            kl_coeff=tune.uniform(0.3, 1),
            kl_target=tune.uniform(0.003, 0.03),
            vf_loss_coeff=tune.uniform(0.5, 1),
            entropy_coeff=tune.uniform(0, 0.01),
            grad_clip=1,
            grad_clip_by="value",
            _enable_learner_api=False,
        )
    ppo_config.resources(
            num_cpus_per_worker=cpus_per_worker,
            num_cpus_for_local_worker=cpus_for_local_worker,
            placement_strategy="PACK",
        )
    ppo_config.rollouts(num_rollout_workers=rollout_workers)
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.callbacks(ReportModelStateCallback)

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
        stop=CombinedStopper(
            RewardMinStopper(min_reward_threshold=80),
            MaxTimestepsStopper(max_timesteps=max_timesteps),
        ),        
        storage_path=storage_path,
        local_dir=storage_path,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute="episode_reward_min",
            num_to_keep=1,
            checkpoint_frequency=20,
            checkpoint_at_end=True),
    )

    # tune config
    tune_config = tune.TuneConfig(
            num_samples=tune_samples,
            scheduler= ASHAScheduler(
                time_attr='timesteps_total',
                metric='episode_reward_min',
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
    parser.add_argument('--location',               default="local", choices=['cluster', 'local'], help='execution location, setting depending variables')
    parser.add_argument('--actor_config',           default="model_GINE.json", help="path to the actor model config")
    parser.add_argument('--critic_config',          default="model_GAT.json", help="path to the critic model config")
    parser.add_argument('--encoders_config',        default="encoders_sincos.json", help="path to the encoders config")
    parser.add_argument('--env_config',             default="env_comv2_0.json", help="path to env config")
    parser.add_argument('--min_timesteps',          default=15000, type=int, help="min number of min_timesteps to run")
    parser.add_argument('--max_timesteps',          default=500000, type=int, help="max number of max_timesteps to run")
    parser.add_argument('--batch_size',             default=512, type=int, help="batch size for training")
    parser.add_argument('--rollout_workers',        default=0, type=int, help="number of rollout workers")
    parser.add_argument('--cpus_per_worker',        default=1, type=int, help="number of cpus per rollout worker")
    parser.add_argument('--cpus_for_local_worker',  default=2, type=int, help="number of cpus for local worker")
    parser.add_argument('--tune_samples',           default=1000, type=int, help="number of samples to run")
    parser.add_argument('--performance_study',      default=False, action='store_true', help='run performance study with fixed set of parameters and run length')
    parser.add_argument('--ray_threads',            default=None, type=int, help="number of threads to use for ray")
    

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
        min_timesteps=args.min_timesteps,
        max_timesteps=args.max_timesteps,
        batch_size=args.batch_size,
        rollout_workers=args.rollout_workers, 
        cpus_per_worker=args.cpus_per_worker,
        cpus_for_local_worker=args.cpus_for_local_worker,
        tune_samples=args.tune_samples,
        performance_study=args.performance_study,
        ray_threads=args.ray_threads) 



