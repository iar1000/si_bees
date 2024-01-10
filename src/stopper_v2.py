from ray.tune import Stopper

# @todo
class MinEpisodeLengthStopper(Stopper):
        def __init__(self, min_episode_len_mean: int):
            self.min_episode_len_mean = min_episode_len_mean
            self.exit = False

        def __call__(self, trial_id, result):
            self.exit = result["episode_len_mean"] <= self.min_episode_len_mean
            return self.exit
        
        def stop_all(self):
            return self.exit

class RewardMinStopper(Stopper):
        def __init__(self, min_reward_threshold: int):
            self.min_reward_threshold = min_reward_threshold
            self.exit = False

        def __call__(self, trial_id, result):
            self.exit = result["episode_reward_min"] > self.min_reward_threshold
            if self.exit:
                 print(f"\n\n=============  max_min_reward stopper  ====================")
                 print(f"trial {trial_id} reached threshold!\nkill all open trials!")
                 print(f"threshold={self.min_reward_threshold}, achieved={result['episode_reward_min']}")
                 print(f"===========================================================\n\n")
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
    