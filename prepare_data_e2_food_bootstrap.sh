#!/bin/bash

#!/bin/sh
out=/xdisk/data/dstc/xtrack/e2_food_bs
mkdir -p $out
python extra_data_gen.py --out_file $out/train.json \
                         --based_on /xdisk/data/dstc/xtrack/e2_food/train.json

#cp /xdisk/data/dstc/xtrack/e2_food/valid.json $out/valid.json
cp $out/train.json $out/valid.json
#for i in valid; do
#    python extra_data_gen.py --out_file $out/$i.json \
#                             --based_on /xdisk/data/dstc/xtrack/e2_food/$i.json
#done

