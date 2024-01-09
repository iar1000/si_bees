from ray.tune import Stopper


class MinEpisodeLengthStopper(Stopper):
        def __init__(self, min_episode_len_mean: int):
            self.min_episode_len_mean = min_episode_len_mean
            self.exit = False

        def __call__(self, trial_id, result):
            self.exit = result["episode_len_mean"] <= self.min_episode_len_mean
            return self.exit
        
        def stop_all(self):
            return self.exit

class MaxRewardStopper(Stopper):
        def __init__(self, max_reward: int):
            self.max_reward = max_reward
            self.exit = False

        def __call__(self, trial_id, result):
            self.exit = result["episode_reward_mean"] > self.max_reward
            return self.exit
        
        def stop_all(self):
            return self.exit
        
class MaxTimestepsStopper(Stopper):
        def __init__(self, max_timesteps: int):
            self.max_timesteps = max_timesteps

        def __call__(self, trial_id, result):
            return result["num_env_steps_trained"] > self.max_timesteps
        
        def stop_all(self):
            return False
    