

def curriculum_fn(train_results, task_settable_env, env_ctx):
    """expand the distance between the oracle and the plattform if agents have a positive reward"""
    current_task = task_settable_env.get_task()
    reward = train_results["episode_reward_mean"]
    print("HEEEEEELLOO", train_results["custom_metrics"])
    quit()
    optimality = train_results["custom_metrics"]["episode_performance_per_100_mean"]
    if optimality > 0.8:
        return current_task + 1
    else:
        return current_task