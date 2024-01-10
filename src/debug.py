import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.stopper import CombinedStopper
from configs.utils import load_config_dict
from callbacks_v2 import ReportModelStateCallback
from envs.communication_v2.environment import CommunicationV2_env
from envs.communication_v2.models.pyg import GNN_PyG
from stopper_v2 import MaxTimestepsStopper, RewardMinStopper
from utils import create_tunable_config, filter_tunables


config_dir = os.path.join("src", "configs") 
env_config = load_config_dict(os.path.join(config_dir, "env_comv2_0.json"))
logging_config = load_config_dict(os.path.join(config_dir, "logging_local.json"))

actor_config = load_config_dict(os.path.join(config_dir, "model_GINE.json"))
critic_config = load_config_dict(os.path.join(config_dir, "model_GATv2.json"))
encoders = load_config_dict(os.path.join(config_dir, "encoders_sincos.json"))

batch_size = 1024

ray.init(num_cpus=1, local_mode=True)

env = CommunicationV2_env

model = {}
tunable_model_config = {}
tunable_model_config["actor_config"] = create_tunable_config(filter_tunables(actor_config))
tunable_model_config["critic_config"] = create_tunable_config(critic_config)
tunable_model_config["encoders_config"] = create_tunable_config(encoders)
    
env = CommunicationV2_env
model = {"custom_model": GNN_PyG,
        "custom_model_config": tunable_model_config}


ppo_config = (
    PPOConfig()
    .environment(
        env, # @todo: need to build wrapper
        env_config=env_config,
        disable_env_checking=True)
    .training(
        gamma=0.1,
        lr=0.0005,
        grad_clip=1,
        grad_clip_by="value",
        model=model,
        train_batch_size=batch_size, 
        _enable_learner_api=False
    )
    .rl_module(_enable_rl_module_api=False)
    .callbacks(ReportModelStateCallback)
    .multi_agent(count_steps_by="env_steps")
)

run_config = air.RunConfig(
    stop=CombinedStopper(
        RewardMinStopper(min_reward_threshold=9),
        MaxTimestepsStopper(max_timesteps=100000),
    ),
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


