

def curriculum_fn(train_results, task_settable_env, env_ctx):
    """expand the distance between the oracle and the plattform if agents have a positive reward"""
    current_task = task_settable_env.get_task()

    # increase task if it has been solve with 80% accuracy
    if "episode_optimality_capped_mean" in train_results["custom_metrics"].keys() and \
        train_results["custom_metrics"]["episode_optimality_capped_mean"] > 0.8:
        return current_task + 1

    else:
        return current_task