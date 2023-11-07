from typing import Dict, Tuple
import numpy as np

from pprint import pprint

from ray import train
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.callback import Callback
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper


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
        assert all(x==rewards[0] for x in rewards), "assumption that all agent rewards are equal is violated"

        # @info:    if only custom_metrics is kept, rllib automatically calculates min, max and mean from it, but throws the rest
        #           adding it to hist_data keeps the raw episode rewards for post processing
        optimality = 0 if env.model.max_reward <= 0 else env.model.accumulated_reward / env.model.max_reward
        episode.custom_metrics["episode_optimality"] = optimality
        episode.custom_metrics["episode_optimality_capped"] = max(-1, optimality)
        episode.custom_metrics["episode_reward_normalized"] = rewards[0]
        episode.custom_metrics["curriculum_task"] = env.get_task()
        episode.custom_metrics["curr_learning_score"] = int(env.get_task()) * 2 + max(-1, optimality)
