#!/bin/sh
out=/xdisk/data/dstc/xtrack/toy
mkdir -p $out
python toy_gen.py --out_file $out/train.json
for i in test valid; do
    python toy_gen.py --out_file $out/$i.json --vocab_from $out/train.json ;
done
