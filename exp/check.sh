#!/bin/bash

EXPERIMENT_NAME=$(basename "$1")

EXPERIMENT_OUT=exp/results/${EXPERIMENT_NAME}

extra=--print_header

cfg_cnt=0
while read cfg; do
    cfg_cnt=$((cfg_cnt+1))
    for i in 1 2 3 4 5; do
        #echo -n "${i};${cfg};"
        eid=${EXPERIMENT_NAME}_cfg${cfg_cnt}_${i}
        STDERR_LOCATION=${EXPERIMENT_OUT}/${eid}.stderr.txt
        python xtrack_parse_log.py ${STDERR_LOCATION} $extra
        extra=
        #bash exp/check_file.sh ${STDERR_LOCATION}
    done
done < $1
