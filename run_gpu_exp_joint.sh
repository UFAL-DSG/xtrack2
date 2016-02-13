#!/bin/bash
set -x

#. job_pool.sh

#job_pool_init 4 0
TMP_DIR=/a/SSD/zilka/tmp/


for i in $(seq 0 9); do

    E_NAME="${WHOLE_E_NAME}_${i}"

    DATA_BUILD_LOG=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.${i}.csv
    DATA_NAME=$(python build_data_dstc2.py --e_name ${E_NAME} ${DATA_BUILD_FLAGS} 2>&1 | tee  | tail -n 1)

    GPU=$1
    E_DIR=${TMP_DIR}/${E_NAME}
    DATA_DIR="data/xtrack/${DATA_NAME}"

    export THEANO_FLAGS="allow_gc=False,device=gpu${GPU},floatX=float32"
    python xtrack2.py ${DATA_DIR} --override_slots food,area,pricerange --out ${E_DIR}/goals/ --early_stopping_group goals ${XTRACK_FLAGS}
done


E_NAME=${WHOLE_E_NAME}

for dataset in dev test; do
    TRACK_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.json
    TRACK_LOG=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.log
    SCORE_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.csv
    RESULTS_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.results.txt

    echo "--- score_goals_ens.sh" >> ${RESULTS_FILE}
    date >> ${RESULTS_FILE}

    (
        echo "--data_file ${DATA_DIR}/${dataset}.json"
        echo "--track_log ${TRACK_LOG}"

        echo "--output_file ${TRACK_FILE}"

        echo "--params_file /a/SSD/zilka/tmp/${E_NAME}_0/name/params.final.p"
        for i in $(seq 0 9); do
            echo "--params_file /a/SSD/zilka/tmp/${E_NAME}_${i}/goals/params.final.p"
        done

    ) | xargs python dstc_tracker.py

    python dstc_scripts/score.py --dataroot data/dstc2/data/ --dataset dstc2_${dataset} --trackfile ${TRACK_FILE} --scorefile ${SCORE_FILE} --ontology dstc_scripts/config/ontology_dstc2.json

    cat ${SCORE_FILE} | grep -E "(method, acc, 2, a)|(method, l2, 2, a)|(goal.joint, acc, 2, a)|(goal.joint, l2, 2, a)|(requested.all, acc, 2, a)|(requested.all, l2, 2, a)" | tee -a ${RESULTS_FILE}
    echo "Results file: ${RESULTS_FILE}"
done