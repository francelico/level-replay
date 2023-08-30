partition=$1
shift 1
experiment_lists=("$@")

SEEDS=(8 88 888 8888)
batch_script=job_mila_${partition}.sh

cd ~/procgen/level-replay/slurm
SLEEP_TIME=1

for experiment_list in "${experiment_lists[@]}"; do
    NR_EXPTS=`cat ${experiment_list}.txt | wc -l`
    for experiment_no in `seq 1 ${NR_EXPTS}`; do
        for seed in "${SEEDS[@]}"; do
            echo "executing sbatch $batch_script $experiment_list $experiment_no $seed"
            sbatch $batch_script $experiment_list $experiment_no $seed
            sleep ${SLEEP_TIME}
        done
    done
done





