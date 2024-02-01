from typing import Dict
from numpy import mean


from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class SimpleCallback(DefaultCallbacks):

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
        env = base_env.get_sub_environments()[env_index]
        model = env.model
        episode.custom_metrics["reward_total"] = model.reward_total
        episode.custom_metrics["reward_lower_bound"] = model.reward_lower_bound
        episode.custom_metrics["reward_upper_bound"] = model.reward_upper_bound
        episode.custom_metrics["n_state_switches"] = model.n_state_switches
        episode.custom_metrics["reward_percentile"] = (model.reward_total - model.reward_lower_bound) / (model.reward_upper_bound - model.reward_lower_bound) if model.reward_upper_bound - model.reward_lower_bound != 0 else 0
        episode.custom_metrics["reward_score"] = episode.custom_metrics["reward_percentile"] + int(env.get_task())

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ) -> None:
        pass


