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
from models.gnn_base import GNN_ComNet


def run(logging_config: dict, 
        model_config: dict,
        env_config: dict,
        tune_config: dict):
    """starts a run with the given configurations"""

    ray.init()
    
    run_name = env_config["task_name"] + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    storage_path = os.path.join(logging_config["storage_path"], run_name)

    # calculate some configuration data into ray language
    # only used when train_batch_size is not a hyperparameter
    epochs_per_training_batch = tune_config["epochs_per_training_batch"]
    train_batch_size = epochs_per_training_batch * env_config["env_config"]["max_steps"], # ts per iteration
    train_batch_size = train_batch_size[0]
    
    # make checkpoint frequency such that the biggest batch size still gets it, the smaller one produce more
    checkpoint_every_n_timesteps = tune_config["checkpoint_every_n_timesteps"]
    max_training_batch_size = 8192
    min_checkpoint_frequency = int(checkpoint_every_n_timesteps/max_training_batch_size)
    print(f"checkpointing schedule every {min_checkpoint_frequency} iterations")

    # set task environment
    env = CommunicationV0_env
    
    # create internal model from config
    model = {}
    if model_config["model"] == "FullyConnected":
        model = {"custom_model": FullyConnected,
                "custom_model_config": model_config["model_config"]}
    elif model_config["model"] == "GNN_ComNet":
        model = {"custom_model": GNN_ComNet,
                "custom_model_config": model_config["model_config"]}
    nh_size = (2 * env_config["env_config"]["agent_config"]["com_range"] + 1)**2
    model["custom_model_config"]["n_states"] = nh_size + 1 # pass in neighborhood size to calculate per agent space size automatically

    # set config
    ppo_config = (
        PPOConfig()
        .environment(
            env, # @todo: need to build wrapper
            env_config=env_config["env_config"],
            env_task_fn=curriculum_fn if env_config["env_config"]["curriculum_learning"] else NotProvided,
            disable_env_checking=True)
        #.resources(
        #    num_gpus=resources_config["num_gpus"],
        #    )
        .training(
            gamma=tune.uniform(0.1, 0.9),
            lr=tune.uniform(1e-4, 1e-1),
            grad_clip=1,
            grad_clip_by="value",
            model=model,
            train_batch_size=tune.choice([8192, 16384]), # ts per iteration
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
        checkpoint_frequency=min_checkpoint_frequency,
        checkpoint_at_end=True
    )

    run_config = air.RunConfig(
        name=run_name,
        stop={"timesteps_total": tune_config["max_timesteps"]}, # https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        storage_path=storage_path,
        callbacks=callbacks,
        checkpoint_config=checkpoint_config,
    )

    asha_scheduler = ASHAScheduler(
        time_attr='timesteps_total',
        metric='custom_metrics/curr_learning_score_mean',
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
    parser.add_argument('-model_config', default="model_fc.json", help="path to the NN model config")
    parser.add_argument('-env_config', default="env_comv0.json", help="path to task/ env config")
    parser.add_argument('-tune_config', default="tune_ppo.json", help="path to tune config")

    args = parser.parse_args()

    # load configs
    config_dir = os.path.join("src", "configs")
    model_config = load_config_dict(os.path.join(config_dir, args.model_config))
    env_config = load_config_dict(os.path.join(config_dir, args.env_config))
    tune_config = load_config_dict(os.path.join(config_dir, args.tune_config))
    
    # location dependend configs
    if args.location == 'cluster':
        #resources_config = load_config_dict(os.path.join(config_dir, "resources_cluster.json"))
        logging_config = load_config_dict(os.path.join(config_dir, "logging_cluster.json"))
    else:
        #resources_config = load_config_dict(os.path.join(config_dir, "resources_local.json"))
        logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

    # override default configs
    #if args.resources_config:
    #    resources_config = load_config_dict(os.path.join(config_dir, args.resources_config))
    #if args.logging_config:
    #    logging_config = load_config_dict(os.path.join(config_dir, args.logging_config))

    # sanity print
    print("===== run hyperparameter tuning =======")
    for k, v in args.__dict__.items():
        print(f"\t{k}: {v}")
    print("\n")

    run(logging_config=logging_config,
        model_config=model_config,
        env_config=env_config,
        tune_config=tune_config)



