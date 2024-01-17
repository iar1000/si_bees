import ray
from ray import air, tune
from ray.train import CheckpointConfig
from ray.tune.stopper import CombinedStopper
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig
from datetime import datetime

from callback import SimpleCallback
from environment import Simple_env
from pyg import GNN_PyG
from stopper import MaxTimestepsStopper, RewardMinStopper

ray.init()

tune.register_env("Simple_env", lambda env_config: Simple_env(env_config))
model = {"custom_model": GNN_PyG}
# ppo config
ppo_config = PPOConfig()
ppo_config.environment(
        "Simple_env",
        disable_env_checking=True)
ppo_config.training(
        model=model,
        train_batch_size=512,
        #shuffle_sequences=True,
        #use_critic=True,
        #use_gae=True,
        #lambda_=tune.uniform(0.9, 1),
        gamma=0.99,
        lr=tune.uniform(5e-6, 0.003),
        #clip_param=tune.choice([0.1, 0.2, 0.3]),
        #kl_coeff=tune.uniform(0.3, 1),
        #kl_target=tune.uniform(0.003, 0.03),
        #vf_loss_coeff=tune.uniform(0.5, 1),
        #entropy_coeff=tune.uniform(0, 0.01),
        grad_clip=1,
        grad_clip_by="value",
        _enable_learner_api=False)
ppo_config.rollouts(num_rollout_workers=1)
ppo_config.resources(
        num_cpus_per_worker=1,
        num_cpus_for_local_worker=2,
        placement_strategy="PACK")
ppo_config.rl_module(_enable_rl_module_api=False)
ppo_config.callbacks(SimpleCallback)


# run and checkpoint config
run_config = air.RunConfig(
    name=f"simple-env-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    storage_path="/Users/sega/Code/si_bees/log",
    stop=CombinedStopper(
        RewardMinStopper(min_reward_threshold=80),
        MaxTimestepsStopper(max_timesteps=100000),
    ),        
    checkpoint_config=CheckpointConfig(
        checkpoint_score_attribute="episode_reward_min",
        num_to_keep=1,
        checkpoint_frequency=10,
        checkpoint_at_end=True),
)

# tune config
tune_config = tune.TuneConfig(
        num_samples=100000,
        scheduler= ASHAScheduler(
            time_attr='timesteps_total',
            metric='episode_reward_min',
            mode='max',
            grace_period=5000,
            max_t=500000,
            reduction_factor=2)
    )

tuner = tune.Tuner(
    "PPO",
    run_config=run_config,
    tune_config=tune_config,
    param_space=ppo_config.to_dict()
)

tuner.fit()

