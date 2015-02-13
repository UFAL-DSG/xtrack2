#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

E_ROOT=${DATA_DIRECTORY}/xtrack/e2_food
#SLOTS=food,req_food,method
SLOTS=food

echo "> Processing training data."
python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/ \
    --out_dir ${E_ROOT}/train \
    --flist ${DATA_DIRECTORY}/dstc2/scripts/config/dstc2_train.flist

echo "> Processing validation data."
python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/\
    --out_dir ${E_ROOT}/valid \
    --flist ${DATA_DIRECTORY}/dstc2/scripts/config/dstc2_dev.flist

#echo "> Processing testing data."
#python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/test \
#    --out_dir ${E_ROOT}/test

echo "> Converting data to HDF5 format."
python xtrack_data2.py \
        --data_dir ${E_ROOT}/train \
        --out_file ${E_ROOT}/train.json \
        --slots ${SLOTS} \
        --oov_ins_p 0.05 \
        --include_system_utterances \
        --n_best_order 1
for i in valid; do
    python xtrack_data2.py \
        --data_dir ${E_ROOT}/${i} \
        --out_file ${E_ROOT}/${i}.json \
        --vocab_from ${E_ROOT}/train.json \
        --slots ${SLOTS} \
        --oov_ins_p 0.0 \
        --include_system_utterances \
        --n_best_order 1
done

echo "> Finishing up."
cp prepare_data_e2.sh ${E_ROOT}

date > ${E_ROOT}/timestamp.txt
