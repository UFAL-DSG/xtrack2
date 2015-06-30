#!/bin/bash
set -e

E_NAME=$1
DATA_DIR=$2
#"data/xtrack/${E_NAME}_0_tagged_0n1best_nowcn_xtrack/"


for dataset in dev test; do
    TRACK_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.json
    TRACK_LOG=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.log
    SCORE_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.csv
    RESULTS_FILE=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.results.txt

    (
        echo "--data_file ${DATA_DIR}/${dataset}.json"
        echo "--track_log ${TRACK_LOG}"

        echo "--output_file ${TRACK_FILE}"

        echo "--params_file /a/SSD/zilka/tmp/${E_NAME}_0/name/params.final.p"
        for i in $(seq 0 9); do
            for slot in food area pricerange method req_food,req_area,req_pricerange,req_name,req_phone,req_addr,req_postcode,req_signature; do
                echo "--params_file /a/SSD/zilka/tmp/${E_NAME}_${i}/${slot}/params.final.p"
            done
        done

    ) | xargs python dstc_tracker.py

    python dstc_scripts/score.py --dataroot data/dstc2/data/ --dataset dstc2_${dataset} --trackfile ${TRACK_FILE} --scorefile ${SCORE_FILE} --ontology dstc_scripts/config/ontology_dstc2.json

    cat ${SCORE_FILE} | grep -E "(method, acc, 2, a)|(method, l2, 2, a)|(goal.joint, acc, 2, a)|(goal.joint, l2, 2, a)|(requested.all, acc, 2, a)|(requested.all, l2, 2, a)" | tee -a ${RESULTS_FILE}
    echo "Results file: ${RESULTS_FILE}"
done