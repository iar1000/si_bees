# RL/MARL Framework for PyTorch Geometric based Policies
Our methodology leverages graph abstraction and employs Graph Neural Networks (GNNs) for both the actor and a central critic, trained using Proximal Policy Optimization (PPO). We represent the neighborhood of agents as a graph, utilizing the relative distances between them as edge attributes. Once trained, the policy can seamlessly adapt to any number of agents.

We've constructed the framework using Ray and PyTorch Geometric. The policy networks' convolutions are built with PyG modules, ensuring easy interchangeability. Graph construction is facilitated by a structured observation space, simplifying the creation of custom tasks through agent and edge state space definitions. Network creation is automated within this framework.

Key scripts within our framework include train.py, which handles hyperparameter tuning, `environment.py`, responsible for implementing the RLlib environment, and `task_models.py`, which defines task-specific simulations including agents and observation spaces. The GNN architecture, housing both actor and critic networks, is implemented in `gnn.py`.

For model execution, `run.py` is employed. Additionally, for MESA-based models, an automatic web server facilitates model visualization. We've also integrated a Pygame-based simulation for physics-based tasks.

Finally, model evaluation is conducted through scripts prefixed with `xxx_`, initiated by running `eval.py`.

The framework was produced for the Thesis `Scaling MARL with Graph Neural Networks`

## setup
The project leverages Conda and Ray to seamlessly scale up and optimize performance on compute clusters. Comprehensive instructions for installing and configuring Conda can be found in the ITET [ITET Cluster Handbook](https://hackmd.io/hYACdY2aR1-F3nRdU8q5dA), tailored specifically for ITET members.

All dependencies are installed by creating the following conda environment, or by using `requirements.txt`:
`CONDA_OVERRIDE_CUDA=11.7 conda create --name swarm python pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 pyarrow pandas gymnasium ray-all==2.7.1 pyg==2.4.0 mesa lz4==4.3.2 dm-tree==0.1.8 scikit-image==0.22.0 wandb==0.15.12 -c pytorch -c nvidia -c conda-forge -c pyg`

## implemented tasks
### Lever Pulling
The lever-pulling task, as introduced by Sukhbaatar et al. (2016), serves to evaluate agents' communication skills. We replicated this task within our environment, maintaining its original setup. In a grid-world environment, five out of 500 workers are placed on the same square, forming a fully connected graph where the distance between any two workers is zero. The goal for the workers is to synchronously select five distinct levers.

### Transmission
The transmission task explores the fundamental information propagation capabilities among agents. A group of stationary workers surrounds an oracle, which selects a number for the workers to replicate (denoted as $out$, where $out \in {0,1,2,3,4}$). Each worker has a communication link to the oracle, either directly or through intermediate workers acting as transmitters. Agents must actively relay information to solve the task. The worker's state includes its type, current output, and a hidden vector, while the oracle's hidden vector is fixed at 0, allowing workers to differentiate between other workers and the oracle.

### Transmission Extended
In this variation, workers' states include their relative position with respect to the oracle ($pos$), investigating the impact of knowing absolute positions versus solely having knowledge of local relative positions through edge attributes.

### Moving
Building upon the Transmission task, movement is introduced. In addition to replicating the oracle's output, workers must navigate the environment. Each worker can move one square horizontally and one square vertically in any direction per time step. Initially positioned in a communication line, workers tend to disconnect during early training, resulting in a fragmented graph. To mitigate excessive movement, we employ discretization techniques, favoring idle states during random movement sampling.

### Moving History
Similar to the Transmission Extended task, worker states are augmented with a global reference point, incorporating their past three locations and movements. This enhancement draws parallels to natural agents relying on central bases.

### MPE Spread
Inspired by the Multi-Agent Particle Environment, workers aim to approach landmarks without colliding. The neighborhood function always includes the three closest workers and landmarks. While landmarks remain stationary, workers move continuously, with their state comprising absolute position and velocity. Their actions influence velocity through a physics engine.

### MPE Spread Memory
Worker states are supplemented with a hidden vector, allowing retention of information from previous interactions.

## docs and bugs
https://hackmd.io/hYACdY2aR1-F3nRdU8q5dA  
https://github.com/wandb/wandb/issues/3911  
https://github.com/pytorch/pytorch/issues/101850  
https://github.com/pytorch/pytorch/issues/99625  