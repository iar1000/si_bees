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

# Send some noteworthy information to the output log
echo ""
echo "=== Start slurm scipt ==="
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

NUM_RAY_THREADS=30

# execute script
echo "-> current directory $(pwd)"
echo "-> run train.py from directory $RUN_DIR/$SRC_DIR"
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 1 \
                                        --num_rollouts 0
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 1 \
                                        --num_rollouts 1
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 1 \
                                        --num_rollouts 2
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 1 \
                                        --num_rollouts 4
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 2 \
                                        --num_rollouts 0
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 2 \
                                        --num_rollouts 1
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 2 \
                                        --num_rollouts 2
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 2 \
                                        --num_rollouts 4
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 4 \
                                        --num_rollouts 0
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 4 \
                                        --num_rollouts 1
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 4 \
                                        --num_rollouts 2
python $RUN_DIR/$SRC_DIR/performance.py --project_dir $PROJECT_DIR \
                                        --num_ray_threads $NUM_RAY_THREADS \
                                        --num_cpu_for_local 4 \
                                        --num_rollouts 4


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

