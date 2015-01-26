#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

E_ROOT=${DATA_DIRECTORY}/xtrack/e2
MAX_DECODING_STEPS=3
MAX_LEN=100
MAX_LABELS_IN_DIALOG=15

echo "> Processing training data."
python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/train \
    --out_dir ${E_ROOT}/train

echo "> Processing validation data."
python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/valid \
    --out_dir ${E_ROOT}/valid

#echo "> Processing testing data."
#python import_dstc.py --data_dir ${DATA_DIRECTORY}/dstc2/data/test \
#    --out_dir ${E_ROOT}/test

echo "> Converting data to HDF5 format."
python xtrack_data2.py \
        --data_dir ${E_ROOT}/train \
        --out_file ${E_ROOT}/train.json \
        --slots food,area,pricerange,name
for i in valid; do
    python xtrack_data2.py \
        --data_dir ${E_ROOT}/${i} \
        --out_file ${E_ROOT}/${i}.json \
        --vocab_from ${E_ROOT}/train.json \
        --slots food,area,pricerange,name
done

echo "> Finishing up."
cp prepare_data_e2.sh ${E_ROOT}

date > ${E_ROOT}/timestamp.txt
