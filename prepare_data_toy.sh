#!/bin/sh
out=/xdisk/data/dstc/xtrack/toy
mkdir -p $out
 for i in train test valid; do python toy_gen.py --out_file $out/$i.json ; done
