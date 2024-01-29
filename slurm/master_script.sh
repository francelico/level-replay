partition=$1
shift 1
experiment_lists=("$@")

JOBNAME="procgen"
MAXJOBS=4
JOBCOUNT=0
batch_script=job_${partition}.sh

cd ~/procgen/level-replay/slurm
SLEEP_TIME=1

for experiment_list in "${experiment_lists[@]}"; do
    NR_EXPTS=`cat ${experiment_list}.txt | wc -l`
    for experiment_no in `seq 1 ${NR_EXPTS}`; do
      while [ $JOBCOUNT -gt $MAXJOBS ]; do
        sleep 60
        JOBCOUNT=`squeue -u $USER -h -t pending,running -n $JOBNAME -r | wc -l`
        done
      echo "executing sbatch $batch_script $experiment_list $experiment_no"
      sbatch --job-name=$JOBNAME $batch_script $experiment_list $experiment_no
      sleep ${SLEEP_TIME}
      JOBCOUNT=`squeue -u $USER -h -t pending,running -n $JOBNAME -r | wc -l`
    done
done





