#!/bin/bash

#SBATCH --cpus-per-task=36
#SBATCH --mail-type END
#SBATCH --time=2-00:00:00

# env variables
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

# Send some noteworthy information to the output log
echo ""
echo "=== Start slurm scipt ==="
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# create a temporary run directory
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

# copy repository into the run directory
# note: this step is important, as ray threads fetch the run script over and over from this directory
#       we ensure like this that we always fetch the same code, and not using the "main" repository which can change during the lifetime of the script
echo "-> copy src to ${RUN_DIR}"
cp -r "$PROJECT_DIR/$SRC_DIR" "$RUN_DIR/$SRC_DIR"

# activate conda
[[ -f $CONDA_BIN ]] && eval "$($CONDA_BIN shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "-> conda_env ${CONDA_ENVIRONMENT} activated"
cd ${PROJECT_DIR}

# read in user values
CP_PATH=""
NUM_RAY_THREADS=36
NUM_SAMPLES=10 
NUM_WORKERS=4 
FLAGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cp_path)
      if [[ -n $2 ]]; then
        CP_PATH=$2
        shift 2
      else
        echo "Error: Missing value for -cp_path flag."
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
    --num_samples)
      if [[ -n $2 ]]; then
        NUM_SAMPLES=$2
        shift 2
      else
        echo "Error: Missing value for -num_samples flag."
        exit 1
      fi
      ;;
    --num_workers)
      if [[ -n $2 ]]; then
        NUM_WORKERS=$2
        shift 2
      else
        echo "Error: Missing value for -num_workers flag."
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
echo "    CP_PATH                 = $CP_PATH"
echo "    NUM_RAY_THREADS         = $NUM_RAY_THREADS"
echo "    NUM_SAMPLES             = $NUM_SAMPLES"
echo "    NUM_WORKERS             = $NUM_WORKERS"
echo "    FLAGS                   = $FLAGS"

# Binary or script to execute
echo "-> run train.py from directory $(pwd)"
python $RUN_DIR/$SRC_DIR/train_marl_checkpoint.py --cp_path $CP_PATH --num_ray_threads $NUM_RAY_THREADS --num_samples $NUM_SAMPLES --num_workers $NUM_WORKERS $FLAGS

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

