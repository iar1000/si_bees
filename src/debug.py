import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from configs.utils import load_config_dict
from envs.communication_v0.models.fully_connected import FullyConnected
from envs.communication_v0.models.gnn_base import GNN_ComNet
from callbacks import ReportModelStateCallback
from envs.communication_v1.environment import CommunicationV1_env
from envs.communication_v1.models.pyg import GNN_PyG


config_dir = os.path.join("src", "configs") 
env_config = load_config_dict(os.path.join(config_dir, "env_comv1.json"))
tune_config = load_config_dict(os.path.join(config_dir, "tune_ppo.json"))
logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

ray.init(num_cpus=1, local_mode=True)

env = CommunicationV1_env

# create internal model from config
fc_config = load_config_dict(os.path.join(config_dir, "model_fc.json")) 
model_fc = {"custom_model": FullyConnected,
        "custom_model_config": fc_config["model_config"]}

gnn_config = load_config_dict(os.path.join(config_dir, "model_gnn_comnet.json")) 
model_gnn = {"custom_model": GNN_ComNet,
        "custom_model_config": gnn_config["model_config"]}

gnn_pyg_config = load_config_dict(os.path.join(config_dir, "model_gnn_gcn_fc.json")) 
model_gnn_pyg = {"custom_model": GNN_PyG,
        "custom_model_config": gnn_pyg_config}
model_gnn_pyg["custom_model_config"]["n_agents"] = env_config["env_config"]["agent_config"]["n_agents"]


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
        model=model_gnn_pyg,
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


