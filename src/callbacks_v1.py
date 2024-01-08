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
        env = base_env.get_sub_environments()[env_index]
        
        rewards = [episode.agent_rewards[a] for a in episode.agent_rewards.keys()]
        episode.custom_metrics["reward_normalized"] = rewards[0]
        episode.custom_metrics["max_total_reward"] = env.model.max_total_reward
        episode.custom_metrics["max_obtainable_reward"] = env.model.max_obtainable_reward
        episode.custom_metrics["accumulated_reward"] = env.model.accumulated_reward
        episode.custom_metrics["accumulated_obtainable_reward"] = env.model.accumulated_obtainable_reward

        episode.custom_metrics["curriculum_task"] = env.get_task()

        obtainable_optimality = 0 if env.model.max_obtainable_reward <= 0 else env.model.accumulated_obtainable_reward / env.model.max_obtainable_reward
        episode.custom_metrics["obtainable_optimality"] = obtainable_optimality
        episode.custom_metrics["obtainable_optimality_capped"] = max(-1, obtainable_optimality)
        episode.custom_metrics["obtainable_learning_score"] = int(env.get_task()) * 2 + max(-1, obtainable_optimality)
        
        total_optimality = 0 if env.model.max_total_reward <= 0 else env.model.accumulated_reward / env.model.max_total_reward
        episode.custom_metrics["total_optimality"] = total_optimality
        episode.custom_metrics["total_optimality_capped"] = max(-1, total_optimality)
        episode.custom_metrics["total_learning_score"] = int(env.get_task()) * 2 + max(-1, total_optimality)
        
