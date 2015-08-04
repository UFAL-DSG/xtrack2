#!/bin/bash
set -x

#. job_pool.sh

#job_pool_init 4 0
TMP_DIR=/a/SSD/zilka/tmp/


for slot in food area pricerange name method req_food req_area req_pricerange req_name req_phone req_addr req_postcode req_signature; do
    for i in $(seq 0 9); do

        E_NAME="${WHOLE_E_NAME}_${i}"

        DATA_BUILD_LOG=/a/SSD/zilka/tmp/${E_NAME}.${dataset}.goals.${i}.csv
        DATA_NAME=$(python build_data_dstc2.py --e_name ${E_NAME} ${DATA_BUILD_FLAGS} 2>&1 | tee  | tail -n 1)

        GPU=$1
        E_DIR=${TMP_DIR}/${E_NAME}
        DATA_DIR="data/xtrack/${DATA_NAME}"

        export THEANO_FLAGS="allow_gc=False,device=gpu${GPU},floatX=float32"
        python xtrack2.py ${DATA_DIR} --override_slots ${slot} --out ${E_DIR}/${slot}/ --early_stopping_group ${slot} ${XTRACK_FLAGS}

        if [[ "$slot" != "food" && "$slot" != "area" && "$slot" != "pricerange" ]]; then
            break
        fi
    done
done

bash score_goals_ens.sh ${WHOLE_E_NAME} ${DATA_DIR}
bash score_goals.sh ${WHOLE_E_NAME} ${DATA_DIR}