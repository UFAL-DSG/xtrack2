#!/bin/bash

EXPERIMENT_NAME=$(basename "$1")

EXPERIMENT_OUT=out/
#EXPERIMENT_OUT=exp/results/${EXPERIMENT_NAME}

extra=--print_header

cfg_cnt=0
while read cfg; do
    cfg_cnt=$((cfg_cnt+1))
    for i in $(seq 1 $2); do
        #echo -n "${i};${cfg};"
        eid=${EXPERIMENT_NAME}_cfg${cfg_cnt}_${i}
        STDERR_LOCATION=${EXPERIMENT_OUT}/${eid}_1/log.txt
        python xtrack_parse_log.py ${STDERR_LOCATION} $extra $3
        extra=
        #bash exp/check_file.sh ${STDERR_LOCATION}
    done
done < $1
