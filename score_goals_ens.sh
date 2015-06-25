#!/bin/bash

#!/bin/bash

E_NAME=$1
DATA_DIR="data/xtrack/${E_NAME}_0_tagged_0n1best_nowcn_xtrack/"


for dataset in dev test; do
    TRACK_FILE=/tmp/${E_NAME}.${dataset}.goals.json
    TRACK_LOG=/tmp/${E_NAME}.${dataset}.goals.log
    SCORE_FILE=/tmp/${E_NAME}.${dataset}.goals.csv

    python dstc_tracker.py \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_1/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_2/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_3/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_4/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_5/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_6/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_7/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_8/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_9/food/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_0/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_1/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_2/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_3/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_4/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_5/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_6/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_7/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_8/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_9/area/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_0/name/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_0/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_1/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_2/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_3/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_4/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_5/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_6/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_7/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_8/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_9/pricerange/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_0/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_1/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_2/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_3/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_4/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_5/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_6/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_7/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_8/method/params.final.p \
         --params_file /a/SSD/zilka/tmp/${E_NAME}_9/method/params.final.p \
         --data_file ${DATA_DIR}/${dataset}.json \
         --track_log ${TRACK_LOG} \
         --output_file ${TRACK_FILE}

    python dstc_scripts/score.py --dataroot data/dstc2/data/ --dataset dstc2_${dataset} --trackfile ${TRACK_FILE} --scorefile ${SCORE_FILE} --ontology dstc_scripts/config/ontology_dstc2.json

    echo "############### $dataset"
    cat ${SCORE_FILE} | grep -E "(method, acc, 2, a)|(method, l2, 2, a)|(goal.joint, acc, 2, a)|(goal.joint, l2, 2, a)"

    echo "Score file: ${SCORE_FILE}"
done