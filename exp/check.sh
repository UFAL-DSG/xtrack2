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
        ACC=$(cat ${STDERR_LOCATION} | grep "Tracking accuracy" | grep "valid" | cut -c 59-60 | sort -n | tail -n 1)
        EPOCH=$(cat ${STDERR_LOCATION} | grep "Epoch" | tail -n 1 | cut -d" " -f 6)
        RES="  "
        if [ "$(tail -n 4 ${STDERR_LOCATION} | grep EPILOG | wc -l)" == "0" ]; then
            RES=".."
        else
            if [ "$(tail -n 5 ${STDERR_LOCATION} | grep 'Saving final model' | wc -l)" == "0" ]; then
                RES="ff"
            else
                ACC=$(cat ${STDERR_LOCATION} | grep "Tracking accuracy" | grep "valid" | cut -c 59-60 | sort -n | tail -n 1)
                RES="ok"
            fi
        fi
        echo "${RES} epoch(${EPOCH}) $eid acc(${ACC}) ${STDERR_LOCATION}"
    done
done < $1
