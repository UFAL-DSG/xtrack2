#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

E_ROOT=${DATA_DIRECTORY}/xtrack/e2x${2}
SLOTS="food=food:area=area:pricerange=pricerange:name=name:goals=food,area,pricerange,name:method=method:req=req_food,req_area,req_pricerange,req_name,req_phone,req_addr,req_postcode,req_signature"

if [ "$1" != "skip" ]; then
    echo "> Processing training data."
    python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/ \
        --out_dir ${E_ROOT}/train \
        --use_stringified_system_acts \
        --flist ${DATA_DIRECTORY}/dstc2/scripts/config/dstc2_train${2}.flist

    echo "> Processing validation data."
    python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/\
        --out_dir ${E_ROOT}/valid \
        --use_stringified_system_acts \
        --flist ${DATA_DIRECTORY}/dstc2/scripts/config/dstc2_dev.flist

    echo "> Processing testing data."
    python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/\
        --out_dir ${E_ROOT}/test \
        --use_stringified_system_acts \
        --flist ${DATA_DIRECTORY}/dstc2/scripts/config/dstc2_test.flist
fi

#echo "> Processing testing data."
#python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/test \
#    --out_dir ${E_ROOT}/test

echo "> Building JSON."
python xtrack_data2.py \
        --data_dir ${E_ROOT}/train \
        --out_file ${E_ROOT}/train.json \
        --slots ${SLOTS} \
        --oov_ins_p 0.05 \
        --n_best_order 1 \
        --n_nbest_samples 1 \
        --include_system_utterances \
        --dump_text ${E_ROOT}/train_text.txt

for i in valid test; do
    python xtrack_data2.py \
        --data_dir ${E_ROOT}/${i} \
        --out_file ${E_ROOT}/${i}.json \
        --based_on ${E_ROOT}/train.json \
        --slots ${SLOTS} \
        --oov_ins_p 0.0 \
        --include_system_utterances \
        --n_best_order 1 \
        --n_nbest_samples 1
done

echo "> Finishing up."
cp prepare_data_e2.sh ${E_ROOT}

date > ${E_ROOT}/timestamp.txt

