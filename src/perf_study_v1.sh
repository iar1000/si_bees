#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00

ETH_USERNAME=kpius
PROJECT_NAME=si_bees
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=swarm
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Send some noteworthy information to the output log
echo ""
echo "=== start performance study on $(hostname) ==="
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Set default values for the new variables
ray_threads=20
rollout_workers=0
cpus_per_worker=1
cpus_for_local_worker=1
batch_size_episodes=10
min_episodes=100
tune_samples=1000

# check for user flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ray_threads)
      if [[ -n $2 ]]; then
        ray_threads=$2
        shift 2
      else
        echo "Error: Missing value for --ray_threads flag."
        exit 1
      fi
      ;;
    --rollout_workers)
      if [[ -n $2 ]]; then
        rollout_workers=$2
        shift 2
      else
        echo "Error: Missing value for --rollout_workers flag."
        exit 1
      fi
      ;;
    --cpus_per_worker)
      if [[ -n $2 ]]; then
        cpus_per_worker=$2
        shift 2
      else
        echo "Error: Missing value for --cpus_per_worker flag."
        exit 1
      fi
      ;;
    --cpus_for_local_worker)
      if [[ -n $2 ]]; then
        cpus_for_local_worker=$2
        shift 2
      else
        echo "Error: Missing value for --cpus_for_local_worker flag."
        exit 1
      fi
      ;;
    --batch_size_episodes)
      if [[ -n $2 ]]; then
        batch_size_episodes=$2
        shift 2
      else
        echo "Error: Missing value for --batch_size_episodes flag."
        exit 1
      fi
      ;;
    --min_episodes)
      if [[ -n $2 ]]; then
        min_episodes=$2
        shift 2
      else
        echo "Error: Missing value for --min_episodes flag."
        exit 1
      fi
      ;;
    --tune_samples)
      if [[ -n $2 ]]; then
        tune_samples=$2
        shift 2
      else
        echo "Error: Missing value for --tune_samples flag."
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
echo "RAY_THREADS:           $ray_threads"
echo "ROLLOUT_WORKERS:       $rollout_workers"
echo "CPUS_PER_WORKER:       $cpus_per_worker"
echo "CPUS_FOR_LOCAL_WORKER: $cpus_for_local_worker"
echo "min_episodes: $min_episodes"
echo "batch_size_episodes: $batch_size_episodes"


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
python /itet-stor/kpius/net_scratch/si_bees/src/train.py --location "cluster" --performance_study --ray_threads $ray_threads --rollout_workers $rollout_workers --cpus_per_worker $cpus_per_worker --cpus_for_local_worker $cpus_for_local_worker --min_episodes $min_episodes --batch_size_episodes $batch_size_episodes --tune_samples $tune_samples

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

