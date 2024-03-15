#!/bin/bash

#SBATCH --cpus-per-task=36
#SBATCH --mail-type END
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

# Set a directory for temporary files unique to the job with automatic removal at job termination
#TMPDIR=$(mktemp -d "/itet-stor/kpius/net_scratch/XXXXXXXX")
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

# read in user values
ENV_CONFIG=""
ACTOR_CONFIG="model_GINE.yaml"
CRITIC_CONFIG="model_GATv2.yaml"
NUM_RAY_THREADS=36
NUM_CPU_LOCAL_WORKER=2 
RESTORE_EXPERIMENT_PATH="-"
FLAGS=""
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
    --num_ray_threads)
      if [[ -n $2 ]]; then
        NUM_RAY_THREADS=$2
        shift 2
      else
        echo "Error: Missing value for -num_ray_threads flag."
        exit 1
      fi
      ;;
    --num_cpu_for_local)
      if [[ -n $2 ]]; then
        NUM_CPU_LOCAL_WORKER=$2
        shift 2
      else
        echo "Error: Missing value for -num_cpu_for_local flag."
        exit 1
      fi
      ;;
    --restore)
      if [[ -n $2 ]]; then
        RESTORE_EXPERIMENT_PATH=$2
        FLAGS="$FLAGS --restore $RESTORE_EXPERIMENT_PATH"
        shift 2
      else
        echo "Error: Missing value for -restore flag."
        exit 1
      fi
      ;;
    --enable_gpu)
      FLAGS="$FLAGS --enable_gpu"
      shift 1  # No value needed, just shift by 1
      ;;
    *)
      shift
      ;;
  esac
done

echo "-> user parameters:"
echo "    ENV_CONFIG              = $ENV_CONFIG"
echo "    ACTOR_CONFIG            = $ACTOR_CONFIG"
echo "    CRITIC_CONFIG           = $CRITIC_CONFIG"
echo "    NUM_RAY_THREADS         = $NUM_RAY_THREADS"
echo "    NUM_CPU_LOCAL_WORKER    = $NUM_CPU_LOCAL_WORKER"
echo "    RESTORE_EXPERIMENT_PATH = $RESTORE_EXPERIMENT_PATH"
echo "    FLAGS                   = $FLAGS"

# Binary or script to execute
echo "-> run train.py from directory $(pwd)"
python /itet-stor/kpius/net_scratch/si_bees/src_v2/train.py --env_config $ENV_CONFIG --actor_config $ACTOR_CONFIG --critic_config $CRITIC_CONFIG --num_ray_threads $NUM_RAY_THREADS $FLAGS --num_cpu_for_local $NUM_CPU_LOCAL_WORKER

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
