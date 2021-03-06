#!/bin/bash

echo "$1 --enable_branch_exp" > exp/branch
echo "$1" >> exp/branch

N_EXPERIMENTS=10
branch_name=$(git symbolic-ref -q HEAD)
branch_name=${branch_name##refs/heads/}
branch_name=${branch_name:-HEAD}
EXPERIMENT_NAME="xtrack2.${branch_name}"
SMP=${SMP:-2}
MEM_PER_WORKER=${MEM_PER_WORKER:-4}

EXPERIMENT_OUT=exp/results/${EXPERIMENT_NAME}
mkdir -p ${EXPERIMENT_OUT}

cfg_cnt=0
while read cfg; do
    cfg_cnt=$((cfg_cnt+1))
    for i in $(seq 1 ${N_EXPERIMENTS}); do
        eid=${EXPERIMENT_NAME}_cfg${cfg_cnt}_${i}
        echo Submitting $eid

        STDOUT_LOCATION=${EXPERIMENT_OUT}/${eid}.stdout.txt
        STDERR_LOCATION=${EXPERIMENT_OUT}/${eid}.stderr.txt

        RUN_CMD="qsub -m abe -M lukas@zilka.me -hard -l mem_free=${MEM_PER_WORKER}G,act_mem_free=${MEM_PER_WORKER}G,h_vmem=${MEM_PER_WORKER}G -pe smp ${SMP} -N ${EXPERIMENT_NAME} -o ${STDOUT_LOCATION} -e ${STDERR_LOCATION} -cwd"

        #RUN_CMD="cat"

        $RUN_CMD << ENDOFSCRIPT
        PYTHONUSERBASE=/home/zilka/.local OMP_NUM_THREADS=${SMP} THEANO_FLAGS="base_compiledir=out/${eid}" python xtrack2.py --out out/${eid} --rebuild_model --model_file out/${eid}.model ${cfg}
ENDOFSCRIPT
    done
done < exp/branch
