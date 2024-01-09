from typing import Dict


from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class ReportModelStateCallback(DefaultCallbacks):

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # env = base_env.get_sub_environments()[env_index]
        # episode.custom_metrics["n_steps"] = env.model.curr_step
        pass