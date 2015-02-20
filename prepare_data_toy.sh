#!/bin/sh
out=/xdisk/data/dstc/xtrack/toy
mkdir -p $out
python toy_gen.py --out_file $out/train.json
python toy_gen.py \
    --out_file $out/train.json \
    --based_on $out/train.json \
    --input_len 1 \
    --dialog_len 1
for i in test valid; do
    python toy_gen.py \
        --out_file $out/$i.json \
        --based_on $out/train.json \
        --input_len 1 \
        --dialog_len 1 \
        --dialog_cnt 1000;
done
