import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from envs.communication_v0.environment import CommunicationV0_env 
from experiments import default_config
from models.fully_connected import FullyConnected
from envs.communication_v0.callbacks import ReportModelStateCallback


WANDB_KEY = ".wandb_key"
LOG_FOLDER = "/Users/sega/Code/si_bees/log"

# Limit number of cores
ray.init(num_cpus=2)

task_name = "communication_v0"
ppo_config = (
    PPOConfig()
    .environment(
        CommunicationV0_env, # @todo: need to build wrapper
        env_config=default_config,
        disable_env_checking=True
        # env_task_fn= @todo: curriculum learning
    )
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=1, 
              num_envs_per_worker=1,
              create_env_on_local_worker=True)
    .training(
        train_batch_size=tune.choice([128, 256, 512, 1024, 2048, 4096]),
        gamma=tune.uniform(0.1, 1),
        lr=tune.loguniform(1e-4, 1e-1),
        grad_clip=tune.loguniform(0.1, 50),
        #train_batch_size=tune.randint(1_000, 10_000),
        model={
            "custom_model": FullyConnected,
            "custom_model_config": default_config["model_config"]},
        _enable_learner_api=False
    )
    .rl_module(_enable_rl_module_api=False)
    .callbacks(ReportModelStateCallback)
)


callbacks = list()
wandb_callback = WandbLoggerCallback(
    project=default_config["wandb_project"],
    group=f"tune_{task_name}",
    log_config=True,
    upload_checkpoints=True,
    api_key_file=WANDB_KEY
)
if default_config["enable_wandb"]:
    callbacks.append(wandb_callback)

run_config = air.RunConfig(
    name=task_name,
    # stopping criteria can be everything from here: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
    stop={
        "timesteps_total": default_config["stopper_training_steps_total"]},
    storage_path=LOG_FOLDER,
    callbacks=callbacks
)

tune_config = tune.TuneConfig(
        num_samples=10,
        # @todo: find good parameters
    )

tuner = tune.Tuner(
    "PPO",
    run_config=run_config,
    tune_config=tune_config,
    param_space=ppo_config.to_dict()
)

results = tuner.fit()


