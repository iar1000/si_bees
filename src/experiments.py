

default_config = {
    "task_name": "communication_v0",
    # mesa model
    "mesa_grid_width": 10,
    "mesa_grid_height": 9,
    "mesa_tile_size": 30,
    "mesa_max_rounds": 100,
    "apply_actions_synchronously": True,

    # scenario
    "num_agents": 10,
    "oracle_burn_in": 15,
    "p_oracle_activation": 0.01,

    # agent details
    "n_hidden_vec": 8,
    "n_comm_vec": 10,
    "n_visibility_range": 1,
    "n_comm_range": 1,
    "n_trace_length": 4,

    # tuning
    "tune_stopper_training_iterations": 100,
    "tune_checkpoint_frequency": 0.1,

    # NN
    "model_config": {
            "nn_action_hiddens": [256, 256],
            "nn_action_activation": "relu",
            "nn_value_hiddens": [256, 256],
            "nn_value_activation": "relu"
        },
}


