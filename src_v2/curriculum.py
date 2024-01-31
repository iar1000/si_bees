def curriculum_oracle_switch(train_results, task_settable_env, env_ctx):
    current_task = task_settable_env.get_task()

    if "reward_percentile_mean" in train_results["custom_metrics"].keys() and \
        train_results["custom_metrics"]["reward_percentile_mean"] > 0.75:
        return current_task + 1

    else:
        return current_task