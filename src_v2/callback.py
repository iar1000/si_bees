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
        episode.custom_metrics["ts_to_convergence"] = env.model.ts_to_convergence

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ) -> None:
        ts_to_convergence = result["custom_metrics"]["ts_to_convergence"]
        total_results = len(ts_to_convergence)
        converged_true = [a for a in ts_to_convergence if a >= 0]
        converged_false = [a for a in ts_to_convergence if a < 0]

        result["custom_metrics"]["ts_to_convergence_min"] = min(converged_true) if converged_true else 100
        result["custom_metrics"]["ts_to_convergence_max"] = max(converged_true) if converged_true else 100
        result["custom_metrics"]["ts_to_convergence_mean"] = mean(converged_true) if converged_true else 100
        result["custom_metrics"]["convergence_ratio"] = len(converged_true) / total_results if len(converged_true) and total_results else 0
        result["custom_metrics"]["n_converged_true"] = len(converged_true)
        result["custom_metrics"]["n_converged_false"] = len(converged_false)


