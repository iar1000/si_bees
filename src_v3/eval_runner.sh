#!/bin/bash

ETH_USERNAME=kpius
PROJECT_NAME=si_bees
SRC_DIR="src_v3"
PROJECT_DIR=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
TMP_DIR=/itet-stor/${ETH_USERNAME}/net_scratch
CONDA_BIN=/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda
CONDA_ENVIRONMENT=swarm

mkdir -p ${PROJECT_DIR}/jobs

# Exit on errors
set -o errexit

# activate conda
[[ -f $CONDA_BIN ]] && eval "$($CONDA_BIN shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "-> conda_env ${CONDA_ENVIRONMENT} activated"
cd ${PROJECT_DIR}

# Send some noteworthy information to the output log
echo ""
echo "=== Start slurm scipt ==="
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Define default values for parameters
task_name=""
task_checkpoint=""
task_level=""
eval_episodes=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task_name)
        task_name="$2"
        shift
        shift
        ;;
        --task_checkpoint)
        task_checkpoint="$2"
        shift
        shift
        ;;
        --task_level)
        task_level="$2"
        shift
        shift
        ;;
        --eval_episodes)
        eval_episodes="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# execute script
echo "-> current directory $(pwd)"
echo "-> run eval.py from directory "
python $PROJECT_DIR/src_v3/eval.py --task_name "$task_name" --task_checkpoint "$task_checkpoint" --task_level "$task_level" --eval_episodes "$eval_episodes"

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

