#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

if [ "$1" == "" ]; then
    TRACK_FILE=$(mktemp xtrack_track.XXX.json)
    cp /xdisk/data/dstc/xtrack/e2/valid.flist dstc_scripts/config/xtrack_e2_valid.flist
    python xtrack2_dstc_tracker.py --data_file /xdisk/data/dstc/xtrack/e2/valid.json --output_file ${TRACK_FILE} --model_file $2

    DATASET=xtrack_e2_valid
else
    echo "Using cmdline arguments."
    TRACK_FILE=$1
    DATASET=$2
fi

SCORE_FILE=$(mktemp xtrack_score.XXX.csv)

python dstc_scripts/score.py --dataset ${DATASET} --dataroot /xdisk/data/dstc/dstc2/data --trackfile ${TRACK_FILE} --ontology dstc_scripts/config/ontology_dstc2.json --scorefile ${SCORE_FILE}
python dstc_scripts/report.py --scorefile ${SCORE_FILE}
echo $SCORE_FILE
