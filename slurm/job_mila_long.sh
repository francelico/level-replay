#!/bin/bash
# Author(s): Samuel Garcin (garcin.samuel@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch job_mila.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/mila/s/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/mila/s/%u/slurm_logs/slurm-%A_%a.out

#SBATCH --partition=long

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=30000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=1-00:00:00

# Requeue jobs if they fail
#SBATCH --requeue

# Exclude nodes with known issues
#SBATCH --exclude=cn-g026

experiment=$1
experiment_text_file="${experiment}.txt"
experiment_no=$2
SEED=$3

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

# Make script bail out after first error
set -e

echo "Loading modules"
module load cuda/11.1/cudnn
module load anaconda/3

# Activate your conda environment
CONDA_ENV_NAME=procgen-plr
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

echo "Loading files on node"
REPO_DIR=$HOME/procgen/level-replay
OUT_DIR=$REPO_DIR/results

# logic:
# args: $1 = partition, $2 = array of experiment files \ {.txt}

# partition=$1
# experiment_lists=$2

# batch_script = job_mile_${partition}.sh
# for experiment_list in experiment_lists:
#   NR_EXPTS=`cat ${experiment_list}.txt | wc -l`
#   for experiment_no in `seq 1 ${NR_EXPTS}`; do
#     for seed in seeds:
#       echo "executing sbatch $batch_script $experiment_list $experiment_no $seed"
#       sbatch $batch_script $experiment_list $experiment_no $seed

# each line in experiment_lists.txt should be:
# exp parameters \ { --xpid, --log_dir and --seed --dataset_path --generative_model_checkpoint_path --checkpoint}

LOGDIR=${OUT_DIR}/${experiment}-${experiment_no}
XPID="${experiment}-${experiment_no}-SEED-${SEED}"
echo "Running experiment ${experiment_no} from ${experiment_text_file} with seed ${SEED} and xpid ${XPID}"
echo "Saving logs to ${LOGDIR}"
COMMAND="python ${REPO_DIR}/train.py `sed \"${experiment_no}q;d\" ${experiment_text_file}` --xpid ${XPID} --seed ${SEED} --log_dir ${LOGDIR} --checkpoint"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

# ===================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
