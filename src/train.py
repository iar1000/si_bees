import argparse
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


def run(logging_config: dict, 
        actor_config: dict,
        critic_config: dict,
        env_config: dict,
        tune_config: dict):
    """starts a run with the given configurations"""

    ray.init()
    
    run_name = env_config["task_name"] + "_" + actor_config["model"] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    storage_path = os.path.join(logging_config["storage_path"], run_name)
    local_dir = os.path.join(logging_config["storage_path"], "ray_results")
    train_batch_size = 8192

    # create internal model from config
    def create_tunable_config(config):
        tunable_config = {}
        for k, v in config.items(): 
            if isinstance(v, dict):
                if isinstance(v["min"], int) and isinstance(v["max"], int):
                    tunable_config[k] = tune.choice(list(range(v["min"], v["max"] + 1)))
                else:
                    tunable_config[k] = tune.uniform(v["min"], v["max"])       
            elif isinstance(v, list):
                tunable_config[k] = tune.choice(v)
            else:
                tunable_config[k] = v
        return tunable_config
    
    # set num rounds of actor config to one, as being overriden in a later stage
    def filter_actor_gnn_tunables(config):
        config["gnn_num_rounds"] = 1
        config["gnn_hiddens_size"] = -1
        return config

    env = CommunicationV1_env
    tunable_model_config = {}
    tunable_model_config["actor_config"] = filter_actor_gnn_tunables(create_tunable_config(actor_config))
    tunable_model_config["critic_config"] = create_tunable_config(critic_config)
    model = {"custom_model": GNN_PyG,
            "custom_model_config": tunable_model_config}
    model["custom_model_config"]["n_agents"] = env_config["env_config"]["agent_config"]["n_agents"]

    # ppo config
    ppo_config = (
        PPOConfig()
        .environment(
            env, # @todo: need to build wrapper
            env_config=env_config["env_config"],
            env_task_fn=curriculum_fn if env_config["env_config"]["curriculum_learning"] else NotProvided,
            disable_env_checking=True)
        .training(
            gamma=tune.uniform(0.1, 0.9),
            lr=tune.uniform(1e-4, 1e-1),
            grad_clip=1,
            grad_clip_by="value",
            model=model,
            train_batch_size=train_batch_size,
            _enable_learner_api=False
        )
        .rl_module(_enable_rl_module_api=False)
        .callbacks(ReportModelStateCallback)
        .multi_agent(count_steps_by="env_steps")
    )

    # logging callback
    callbacks = list()
    if logging_config["enable_wandb"]:
        callbacks.append(WandbLoggerCallback(
                            project=logging_config["project"],
                            group=run_name,
                            api_key_file=logging_config["api_key_file"],
                            log_config=logging_config["log_config"],
    ))
        
    # run and checkpoint config
    checkpoint_every_n_timesteps = tune_config["checkpoint_every_n_timesteps"]
    checkpoint_freq = int(checkpoint_every_n_timesteps/train_batch_size)
    run_config = air.RunConfig(
        name=run_name,
        stop={"timesteps_total": tune_config["max_timesteps"]}, # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        storage_path=storage_path,
        local_dir=local_dir,
        callbacks=callbacks,
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=checkpoint_freq,
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
    parser.add_argument('--actor_config', default=None, help="path to the NN model config")
    parser.add_argument('--critic_config', default=None, help="path to the critic model config, only for PyG models")
    parser.add_argument('--env_config', default=None, help="path to task/ env config")
    parser.add_argument('--tune_config', default="tune_ppo.json", help="path to tune config")

    args = parser.parse_args()

    # load configs
    config_dir = os.path.join("src", "configs")
    actor_config = load_config_dict(os.path.join(config_dir, args.actor_config))
    critic_config = load_config_dict(os.path.join(config_dir, args.critic_config))
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
        env_config=env_config,
        tune_config=tune_config)



