import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from configs.utils import load_config_dict
from envs.communication_v0.environment import CommunicationV0_env
from models.fully_connected import FullyConnected
from models.gnn_base import GNN_ComNet
from envs.communication_v0.callbacks import ReportModelStateCallback

config_dir = os.path.join("src", "configs") 
env_config = load_config_dict(os.path.join(config_dir, "env_comv0_1.json"))
tune_config = load_config_dict(os.path.join(config_dir, "tune_ppo.json"))
resources_config = load_config_dict(os.path.join(config_dir, "resources_local.json"))
logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

ray.init(num_cpus=1, local_mode=True)

env = CommunicationV0_env



# create internal model from config
fc_config = load_config_dict(os.path.join(config_dir, "model_fc.json")) 
model_fc = {"custom_model": FullyConnected,
        "custom_model_config": fc_config["model_config"]}

gnn_config = load_config_dict(os.path.join(config_dir, "model_gnn_base.json")) 
model_gnn = {"custom_model": GNN_ComNet,
        "custom_model_config": gnn_config["model_config"]}


ppo_config = (
    PPOConfig()
    .environment(
        env, # @todo: need to build wrapper
        env_config=env_config["env_config"],
        disable_env_checking=True)
    .training(
        gamma=0.1,
        lr=0.0005,
        grad_clip=1,
        grad_clip_by="value",
        model=model_gnn,
        train_batch_size=128, # ts per iteration
        _enable_learner_api=False
    )
    .rl_module(_enable_rl_module_api=False)
    .callbacks(ReportModelStateCallback)
    .multi_agent(count_steps_by="env_steps")
)

run_config = air.RunConfig(
    name="debug",
    stop={"timesteps_total": 100000},
)

tune_config = tune.TuneConfig(
        num_samples=10,
    )

tuner = tune.Tuner(
    "PPO",
    run_config=run_config,
    tune_config=tune_config,
    param_space=ppo_config.to_dict()
)

tuner.fit()


