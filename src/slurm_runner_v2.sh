#!/bin/bash

#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00

ETH_USERNAME=kpius
PROJECT_NAME=si_bees
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=swarm
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo ""
echo "=== Start slurm scipt ==="
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# user argumetns
ENV_CONFIG=""
ACTOR_CONFIG=""
CRITIC_CONFIG=""
ENCODERS_CONFIG=""
RAY_THREADS=""

# check for user flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_config)
      if [[ -n $2 ]]; then
        ENV_CONFIG=$2
        shift 2
      else
        echo "Error: Missing value for -env_config flag."
        exit 1
      fi
      ;;
    --actor_config)
      if [[ -n $2 ]]; then
        ACTOR_CONFIG=$2
        shift 2
      else
        echo "Error: Missing value for -actor_config flag."
        exit 1
      fi
      ;;
    --critic_config)
      if [[ -n $2 ]]; then
        CRITIC_CONFIG=$2
        shift 2
      else
        echo "Error: Missing value for -critic_config flag."
        exit 1
      fi
      ;;
    --encoders_config)
      if [[ -n $2 ]]; then
        ENCODERS_CONFIG=$2
        shift 2
      else
        echo "Error: Missing value for -encoders_config flag."
        exit 1
      fi
      ;;
    --ray_threads)
      if [[ -n $2 ]]; then
        RAY_THREADS=$2
        shift 2
      else
        echo "Error: Missing value for -ray_threads flag."
        exit 1
      fi
      ;;
    *)
      shift
      ;;
  esac
done

echo ""
echo "--- USER ARGUMENTS ---"
echo "ENV_CONFIG:       $ENV_CONFIG"
echo "ACTOR_CONFIG:     $ACTOR_CONFIG"
echo "CRITIC_CONFIG:    $CRITIC_CONFIG"
echo "ENCODERS_CONFIG:  $ENCODERS_CONFIG"
echo "RAY_THREADS:      $RAY_THREADS"

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1
echo ""
echo "-> create and set tmp directory ${TMPDIR}"

# activate conda
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "-> conda_env ${CONDA_ENVIRONMENT} activated"
cd ${DIRECTORY}

# Binary or script to execute
echo "-> run train.py from directory $(pwd)"
python /itet-stor/kpius/net_scratch/si_bees/src/train_v2.py --location "cluster" --env_config $ENV_CONFIG --actor_config $ACTOR_CONFIG --critic_config $CRITIC_CONFIG --encoders_config $ENCODERS_CONFIG --ray_threads $RAY_THREADS

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

