#!/bin/bash

E_NAME=$1
E_DIR=$2
DATA_DIR="data/xtrack/${E_NAME}_tagged_0n1best_nowcn_xtrack"


for dataset in dev test; do
    TRACK_FILE=/tmp/${E_NAME}.${dataset}.req.json
    SCORE_FILE=/tmp/${E_NAME}.${dataset}.req.csv

    # python dstc_tracker.py \
    #     --params_file ${E_DIR}/req_food/params.002.p \
    #     --params_file ${E_DIR}/req_area/params.002.p \
    #     --params_file ${E_DIR}/req_pricerange/params.002.p \
    #     --params_file ${E_DIR}/req_name/params.002.p \
    #     --params_file ${E_DIR}/req_phone/params.002.p \
    #     --params_file ${E_DIR}/req_addr/params.002.p \
    #     --params_file ${E_DIR}/req_postcode/params.002.p \
    #     --params_file ${E_DIR}/req_signature/params.002.p \
    #     --data_file ${DATA_DIR}/${dataset}.json \
    #     --track_log /tmp/${E_NAME}.${dataset}.track.req.txt \
    #     --output_file ${TRACK_FILE}

    python dstc_tracker.py \
        --params_file ${E_DIR}/params.002.p \
        --data_file ${DATA_DIR}/${dataset}.json \
        --track_log /tmp/${E_NAME}.${dataset}.track.req.txt \
        --output_file ${TRACK_FILE}



    python dstc_scripts/score.py --dataroot data/dstc2/data/ --dataset dstc2_${dataset} --trackfile ${TRACK_FILE} --scorefile ${SCORE_FILE} --ontology dstc_scripts/config/ontology_dstc2.json

    cat ${SCORE_FILE} | grep -E "(requested.all, acc, 2, a)|(requested.all, l2, 2, a)"

    echo "Score file: ${SCORE_FILE}"
    echo "Track file: ${TRACK_FILE}"
done