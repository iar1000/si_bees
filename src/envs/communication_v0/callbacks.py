from typing import Dict, Tuple
import numpy as np

from pprint import pprint

from ray.tune.callback import Callback
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class ReportModelStateCallback(DefaultCallbacks):

    def on_sub_environment_created(
        self,
        *,
        worker: "RolloutWorker",
        sub_environment,
        env_context,
        env_index= None,
        **kwargs,
    ) -> None:
        pass

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        episode.hist_data["episode_reward_normalized"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        pass


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
        rewards = [episode.agent_rewards[a] for a in episode.agent_rewards.keys()]
        assert all(x==rewards[0] for x in rewards), "assumption that all agent rewards are equal is violated"

        # @info:    if only custom_metrics is kept, rllib automatically calculates min, max and mean from it, but throws the rest
        #           adding it to hist_data keeps the raw episode rewards for post processing
        episode.hist_data["episode_reward_normalized"].append(rewards[0])
        episode.custom_metrics["episode_reward_normalized"] = rewards[0]


    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        pass
