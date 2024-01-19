from ray.tune import Stopper

class RewardMinStopper(Stopper):
        def __init__(self, min_reward_threshold: int):
            self.min_reward_threshold = min_reward_threshold
            self.exit = False

        def __call__(self, trial_id, result):
            self.exit = result["episode_reward_min"] >= self.min_reward_threshold
            if self.exit:
                 print(f"\n\n=============  max_min_reward stopper  ====================")
                 print(f"trial {trial_id} reached threshold!\nkill all open trials!")
                 print(f"threshold={self.min_reward_threshold}, achieved={result['episode_reward_min']}")
                 print(f"===========================================================\n\n")
            return self.exit
        
        def stop_all(self):
            return self.exit
        
class RewardComboStopper(Stopper):
        def __init__(self, max_reward_threshold: int,
                     mean_reward_threshold: int,
                     min_reward_threshold: int):
            self.max_reward_threshold = max_reward_threshold
            self.mean_reward_threshold = mean_reward_threshold
            self.min_reward_threshold = min_reward_threshold
            self.exit = False

        def __call__(self, trial_id, result):
            hit_max = result["episode_reward_max"] >= self.max_reward_threshold 
            hit_mean = result["episode_reward_mean"] >= self.mean_reward_threshold 
            hit_min = result["episode_reward_min"] >= self.min_reward_threshold 
            self.exit = hit_max and hit_mean and hit_min
            if self.exit:
                 print(f"\n\n=============  mixed reward stopper  ====================")
                 print(f"trial {trial_id} reached threshold!\nkill all open trials!")
                 print(f"threshold max  ={self.max_reward_threshold}, achieved={result['episode_reward_max']}")
                 print(f"threshold mean ={self.mean_reward_threshold}, achieved={result['episode_reward_mean']}")
                 print(f"threshold min  ={self.min_reward_threshold}, achieved={result['episode_reward_min']}")
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
    