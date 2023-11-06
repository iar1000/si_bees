import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.from_config import NotProvided
from ray.train import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from datetime import datetime


from envs.communication_v0.environment import CommunicationV0_env 
from models.fully_connected import FullyConnected
from envs.communication_v0.callbacks import ReportModelStateCallback
from envs.communication_v0.curriculum import curriculum_fn
from configs.utils import load_config_dict


def run(logging_config: dict, 
        resources_config: dict, 
        model_config: dict,
        env_config: dict,
        tune_config: dict):
    """starts a run with the given configurations"""

    ray.init(num_cpus=2)
    
    run_name = env_config["task_name"] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    storage_path = os.path.join(logging_config["storage_path"], run_name)

    # calculate some configuration data into ray language
    epochs_per_training_batch = tune_config["epochs_per_training_batch"]
    checkpoint_every_n_epochs = tune_config["checkpoint_every_n_epochs"]
    
    train_batch_size = epochs_per_training_batch * env_config["env_config"]["max_steps"], # ts per iteration
    train_batch_size = train_batch_size[0]
    checkpoint_frequency = int(checkpoint_every_n_epochs/train_batch_size) # need to convert to iterations


    # set task environment
    env = None
    if env_config["task_name"] == "communication_v0":
        env = CommunicationV0_env
    
    # create internal model from config
    model = {}
    if model_config["model"] == "FullyConnected":
        model = {"custom_model": FullyConnected,
                "custom_model_config": model_config["model_config"]}

    # set config
    ppo_config = (
        PPOConfig()
        .environment(
            env, # @todo: need to build wrapper
            env_config=env_config["env_config"],
            env_task_fn=curriculum_fn if env_config["env_config"]["curriculum_learning"] else NotProvided,
            disable_env_checking=True)
        .resources(num_gpus=resources_config["num_gpus"])
        .rollouts(
            num_rollout_workers=resources_config["num_rollout_workers"], 
            # num_envs_per_worker=resources_config["num_envs_per_worker"]
        )
        .training(
            gamma=tune.uniform(0.1, 0.3),
            lr=tune.uniform(1e-4, 1e-2),
            grad_clip=1,
            model=model,
            train_batch_size=train_batch_size, # ts per iteration
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
        
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_at_end=True
    )

    run_config = air.RunConfig(
        name=env_config["task_name"],
        stop={"timesteps_total": tune_config["max_timesteps"]}, # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        storage_path=storage_path,
        callbacks=callbacks,
        checkpoint_config=checkpoint_config,
    )

    asha_scheduler = ASHAScheduler(
        time_attr='timesteps_total',
        metric='custom_metrics/curriculum_task_max',
        mode='max',
        max_t=tune_config["max_timesteps"],
        grace_period=tune_config["min_timesteps"],
        reduction_factor=2,
    )

    tune_config = tune.TuneConfig(
            num_samples=tune_config["num_samples"],
            scheduler=asha_scheduler
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
    parser.add_argument('-location', default="local", choices=['cluster', 'local'], help='execution location, setting depending variables')
    parser.add_argument('-logging_config', default=None, help="path to the logging config json, defaults to *_local or cluster, depending on location")
    parser.add_argument('-resources_config', default=None, help="path to the available resources config json, defaults to *_local or cluster, depending on location")
    parser.add_argument('-model_config', default="model_fc.json", help="path to the NN model config")
    parser.add_argument('-env_config', default="env_comv0.json", help="path to task/ env config")
    parser.add_argument('-tune_config', default="tune_ppo.json", help="path to tune config")

    args = parser.parse_args()
    print("===== run hyperparameter tuning =======")
    for k, v in args.__dict__.items():
        print(f"\t{k}: {v}")
    print("\n")

    # load configs
    config_dir = os.path.join("src", "configs")
    model_config = load_config_dict(os.path.join(config_dir, args.model_config))
    env_config = load_config_dict(os.path.join(config_dir, args.env_config))
    tune_config = load_config_dict(os.path.join(config_dir, args.tune_config))
    
    # location dependend configs
    if args.location == 'cluster':
        resources_config = load_config_dict(os.path.join(config_dir, "resources_cluster.json"))
        logging_config = load_config_dict(os.path.join(config_dir, "logging_cluster.json"))

    elif args.location == 'local':
        resources_config = load_config_dict(os.path.join(config_dir, "resources_local.json"))
        logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

    # override default configs
    if args.resources_config:
        resources_config = load_config_dict(os.path.join(config_dir, args.resources_config))
    if args.logging_config:
        logging_config = load_config_dict(os.path.join(config_dir, args.logging_config))


    run(logging_config=logging_config,
        resources_config=resources_config,
        model_config=model_config,
        env_config=env_config,
        tune_config=tune_config)



