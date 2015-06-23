#!/bin/bash

#!/bin/bash

E_NAME=$1
E_DIR=$2
DATA_DIR="data/xtrack/${E_NAME}_tagged_0n1best_nowcn_xtrack"


for dataset in dev test; do
    TRACK_FILE=/tmp/${E_NAME}.${dataset}.goals.json
    TRACK_LOG=/tmp/${E_NAME}.${dataset}.goals.log
    SCORE_FILE=/tmp/${E_NAME}.${dataset}.goals.csv

    python dstc_tracker.py \
         --params_file ${E_DIR}/food/params.final.p \
         --params_file ${E_DIR}/area/params.final.p \
         --params_file ${E_DIR}/name/params.final.p \
         --params_file ${E_DIR}/pricerange/params.final.p \
         --params_file ${E_DIR}/method/params.final.p \
         --data_file ${DATA_DIR}/${dataset}.json \
         --track_log ${TRACK_LOG} \
         --output_file ${TRACK_FILE}

    python dstc_scripts/score.py --dataroot data/dstc2/data/ --dataset dstc2_${dataset} --trackfile ${TRACK_FILE} --scorefile ${SCORE_FILE} --ontology dstc_scripts/config/ontology_dstc2.json

    echo "############### $dataset"
    cat ${SCORE_FILE} | grep -E "(method, acc, 2, a)|(method, l2, 2, a)|(goal.joint, acc, 2, a)|(goal.joint, l2, 2, a)"

    echo "Score file: ${SCORE_FILE}"
done