#!/bin/bash

out=/xdisk/data/dstc/xtrack/e2_food_bs1
mkdir -p $out
python extra_data_gen.py --out_file $out/train.json \
                         --based_on /xdisk/data/dstc/xtrack/e2_food/train.json \
                         --easy_dialog_cnt 1 \
                         --switched_dialog_cnt 0


cp $out/train.json $out/valid.json


out=/xdisk/data/dstc/xtrack/e2_food_bs2
mkdir -p $out
python extra_data_gen.py --out_file $out/train.json \
                         --based_on /xdisk/data/dstc/xtrack/e2_food/train.json \
                         --easy_dialog_cnt 500 \
                         --switched_dialog_cnt 0 \
                         --include_base_seqs

#python extra_data_gen.py --out_file $out/valid.json \
#                         --based_on /xdisk/data/dstc/xtrack/e2_food/train
# .json \
#                         --easy_dialog_cnt 10 \
#                         --switched_dialog_cnt 100
cp /xdisk/data/dstc/xtrack/e2_food/valid.json $out/valid.json

#for i in valid; do
#    python extra_data_gen.py --out_file $out/$i.json \
#                             --based_on /xdisk/data/dstc/xtrack/e2_food/$i.json
#done

