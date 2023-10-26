import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils import check_env

from environment import env
from experiments import default_config
from models.fully_connected import FullyConnected

#from action_mask_model import TorchActionMaskModel
# import rllib pettingzoo interface


WANDB_KEY = ".wandb_key"
LOG_FOLDER = "/Users/sega/Code/si_bees/log"

# Limit number of cores
ray.init(num_cpus=2)


# register that way to make the environment under an rllib name
task_name = "communication_v0"
register_env(task_name, lambda config: PettingZooEnv(env(config=config, task=task_name)))

ppo_config = (
    PPOConfig()
    .environment(
        task_name,
        env_config=default_config,
        disable_env_checking=True
        # env_task_fn= @todo: curriculum learning
    )
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    .training(
        gamma=0.9,
        lr=1e-4,
        grad_clip=40.0,
        #train_batch_size=tune.randint(1_000, 10_000),
        model={
            "custom_model": FullyConnected,
            "custom_model_config": default_config["model_config"]},
        _enable_learner_api=False
    )
    .rl_module(_enable_rl_module_api=False)
)

checkpoint_config = air.CheckpointConfig(
    checkpoint_at_end=True,
    checkpoint_frequency=default_config["tune_checkpoint_frequency"] * default_config["tune_stopper_training_iterations"]
)

wandb_callback = WandbLoggerCallback(
    project=f"marl_si_{task_name}",
    log_config=True,
    api_key_file=WANDB_KEY
)

run_config = air.RunConfig(
    name="TODO",
    stop={"training_iteration": default_config["tune_stopper_training_iterations"]},
    checkpoint_config=checkpoint_config,
    callbacks=[wandb_callback],
    storage_path=LOG_FOLDER,
)

tune_config = tune.TuneConfig(
        num_samples=1,
        # @todo: find good parameters
    )

tuner = tune.Tuner(
    "PPO",
    run_config=run_config,
    tune_config=tune_config,
    param_space=ppo_config.to_dict()
)


results = tuner.fit()
results.get_dataframe().to_csv("tune_output.csv")

print("Best hyperparameters found were: ", results.get_best_result(metric="episode_reward_mean", mode="max").config)
