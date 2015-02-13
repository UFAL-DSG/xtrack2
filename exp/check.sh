#!/bin/bash

EXPERIMENT_NAME=$(basename "$1")

EXPERIMENT_OUT=exp/results/${EXPERIMENT_NAME}

cfg_cnt=0
while read cfg; do
    echo $cfg
    cfg_cnt=$((cfg_cnt+1))
    for i in 1 2 3; do
        eid=${EXPERIMENT_NAME}_cfg${cfg_cnt}_${i}
        STDERR_LOCATION=${EXPERIMENT_OUT}/${eid}.stderr.txt
        bash exp/check_file.sh ${STDERR_LOCATION}
    done
done < $1
