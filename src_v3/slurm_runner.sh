#!/bin/bash

#SBATCH --cpus-per-task=36
#SBATCH --mail-type END
#SBATCH --time=2-00:00:00

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

# copy repository into the run directory
# note: this step is important, as ray threads fetch the run script over and over from this directory
#       we ensure like this that we always fetch the same code, and not using the "main" repository which can change during the lifetime of the script
mkdir -p ${TMP_DIR}/tmp
RUN_DIR=$(mktemp -d "$TMP_DIR/tmp/XXXXXXXX")
if [[ ! -d ${TMP_DIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMP_DIR}"' EXIT
export TMP_DIR
echo "-> create temporary run directory ${RUN_DIR}"

# copy repository into the tmp directory
echo "-> copy src to ${RUN_DIR}"
cp -r "$PROJECT_DIR/$SRC_DIR" "$RUN_DIR/$SRC_DIR"

# activate conda
[[ -f $CONDA_BIN ]] && eval "$($CONDA_BIN shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "-> conda_env ${CONDA_ENVIRONMENT} activated"
cd ${PROJECT_DIR}

# read in user values
ENV_CONFIG=""
ACTOR_CONFIG=""
CRITIC_CONFIG=""
ENCODING_CONFIG=""
RESTORE_EXPERIMENT_PATH="-"
NUM_TRIALS=250
MAX_TIMESTEPS=2000000
GRACE_PERIOD=35000
NUM_RAY_THREADS=36
NUM_CPU_LOCAL_WORKER=2 
NUM_ROLLOUTS=0 
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
    --encoding_config)
      if [[ -n $2 ]]; then
        ENCODING_CONFIG=$2
        shift 2
      else
        echo "Error: Missing value for -encoding_config flag."
        exit 1
      fi
      ;;
    --num_trials)
      if [[ -n $2 ]]; then
        NUM_TRIALS=$2
        shift 2
      else
        echo "Error: Missing value for -num_trials flag."
        exit 1
      fi
      ;;
    --max_timesteps)
      if [[ -n $2 ]]; then
        MAX_TIMESTEPS=$2
        shift 2
      else
        echo "Error: Missing value for -max_timesteps flag."
        exit 1
      fi
      ;;
    --grace_period)
      if [[ -n $2 ]]; then
        GRACE_PERIOD=$2
        shift 2
      else
        echo "Error: Missing value for -grace_period flag."
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
    --num_rollouts)
      if [[ -n $2 ]]; then
        NUM_ROLLOUTS=$2
        shift 2
      else
        echo "Error: Missing value for -num_rollouts flag."
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
    *)
      shift
      ;;
  esac
done

# Send some noteworthy information to the output log
echo ""
echo "=== Start slurm scipt ==="
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Script parameters:"
echo "    PROJECT_DIR             = $PROJECT_DIR"
echo "    ENV_CONFIG              = $ENV_CONFIG"
echo "    ACTOR_CONFIG            = $ACTOR_CONFIG"
echo "    CRITIC_CONFIG           = $CRITIC_CONFIG"
echo "    ENCODING_CONFIG         = $ENCODING_CONFIG"
echo "    RESTORE_EXPERIMENT_PATH = $RESTORE_EXPERIMENT_PATH"
echo "    NUM_TRIALS              = $NUM_TRIALS"
echo "    MAX_TIMESTEPS           = $MAX_TIMESTEPS"
echo "    GRACE_PERIOD            = $GRACE_PERIOD"
echo "    NUM_RAY_THREADS         = $NUM_RAY_THREADS"
echo "    NUM_CPU_LOCAL_WORKER    = $NUM_CPU_LOCAL_WORKER"
echo "    NUM_ROLLOUTS            = $NUM_ROLLOUTS"
echo "    FLAGS                   = $FLAGS"

# execute script
echo "-> current directory $(pwd)"
echo "-> run train.py from directory $RUN_DIR/$SRC_DIR"
python $RUN_DIR/$SRC_DIR/train.py       --project_dir $PROJECT_DIR \
                                        --env_config $ENV_CONFIG \
                                        --actor_config $ACTOR_CONFIG \
                                        --critic_config $CRITIC_CONFIG \
                                        --encoding_config $ENCODING_CONFIG \
                                        --num_trials $NUM_TRIALS \
                                        --max_timesteps $MAX_TIMESTEPS \
                                        --grace_period $GRACE_PERIOD \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local $NUM_CPU_LOCAL_WORKER \
                                        --num_rollouts $NUM_ROLLOUTS
                                        $FLAGS \

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

