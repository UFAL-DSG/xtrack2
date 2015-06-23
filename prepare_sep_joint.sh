#!/bin/bash

#. job_pool.sh

#job_pool_init 4 0


for i in $(seq 0 9); do

    E_NAME="e096_asru5_${i}"

    #for slot in method req_food,
    #            # req_area, req_pricerange, req_name, req_phone,
    #            # req_addr, req_postcode, req_signature; do
    #    python build_data_dstc2.py --e_name e093_sjoint --train_nbest_entries 0,1 --only_slot ${slot} --tagged
    #done

    python build_data_dstc2.py --e_name ${E_NAME} --train_nbest_entries 0,1 --tagged

    GPU=1
    E_DIR=/tmp/${E_NAME}
    DATA_DIR="data/xtrack/${E_NAME}_tagged_0n1best_nowcn_xtrack"

    echo edir: ${E_DIR}

    #for slot in req_food req_area req_pricerange req_name req_phone req_addr req_postcode req_signature; do
    for slot in food area pricerange name method ; do # req_food,req_area,req_pricerange,req_name,req_phone,req_addr,req_postcode,req_signature; do
         export THEANO_FLAGS="allow_gc=False,device=gpu${GPU},floatX=float32"
         python xtrack2.py ${DATA_DIR} --lr 0.001 --opt_type adam --mb_size 10 --n_cells 100 --early_stopping_group ${slot} --emb_size 170 --input_n_layers 1 --input_activation rectify --input_n_hidden 300 --rnn_n_layers 1 --wcn_aggreg max --n_early_stopping 1 --ftr_emb_size 50 --input_n_layers 1 --override_slots ${slot} --out ${E_DIR}/${slot}/
    done

    bash score_goals.sh ${E_NAME} ${E_DIR} | tee ${E_DIR}/score.txt
    #bash score_req.sh ${E_NAME} ${E_DIR} | tee -a ${E_DIR}/score.txt

    echo edir: ${E_DIR}
done