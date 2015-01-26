#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

for i in train valid; do
    find -L ${DATA_DIRECTORY}/dstc2/data/train -type d -name "voip*" > dstc_scripts/config/xtrack_e2_${i}.flist
done

TRACK_FILE=$(mktemp xtrack_track.XXX.json)
SCORE_FILE=$(mktemp xtrack_score.XXX.csv)

python xtrack2_dstc_tracker.py --data_file /xdisk/data/dstc/xtrack/e2/train.json --output_file ${TRACK_FILE} --model_file xtrack_model_final.pickle
python dstc_scripts/score.py --dataset xtrack_e2_train --dataroot / --trackfile ${TRACK_FILE} --ontology dstc_scripts/config/ontology_dstc2.json --scorefile ${SCORE_FILE}
python dstc_scripts/report.py --scorefile ${SCORE_FILE}