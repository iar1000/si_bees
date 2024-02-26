from ray.tune import Stopper
        
class max_timesteps_stopper(Stopper):
        def __init__(self, max_timesteps: int):
            self.max_timesteps = max_timesteps

        def __call__(self, trial_id, result):
            return result["num_env_steps_trained"] > self.max_timesteps
        
        def stop_all(self):
            return False
    