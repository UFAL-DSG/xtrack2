#!/bin/bash

EXPERIMENT_NAME=$(basename "$1")
SMP=${SMP:-2}
MEM_PER_WORKER=${MEM_PER_WORKER:-5}

EXPERIMENT_OUT=exp/results/${EXPERIMENT_NAME}
mkdir -p ${EXPERIMENT_OUT}

cfg_cnt=0
while read cfg; do
    cfg_cnt=$((cfg_cnt+1))
    for i in $(seq 1 $2); do
        eid=${EXPERIMENT_NAME}_cfg${cfg_cnt}_${i}
        echo Submitting $eid

        STDOUT_LOCATION=${EXPERIMENT_OUT}/${eid}.stdout.txt
        STDERR_LOCATION=${EXPERIMENT_OUT}/${eid}.stderr.txt
	. qsub_cmd.sh
        #RUN_CMD="cat"

        $RUN_CMD << ENDOFSCRIPT
        PYTHONUSERBASE=/storage/ostrava1/home/ticcky/.local OMP_NUM_THREADS=${SMP} THEANO_FLAGS="base_compiledir=out/${eid},device=cpu" python xtrack2.py --track_log out/${eid}_1/track.log --eid ${eid} --out out/${eid} ${cfg}
ENDOFSCRIPT
    done
done < $1
