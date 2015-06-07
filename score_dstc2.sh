#!/bin/sh
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

. ./config.sh

TRACK_FILE=$1
DATASET=$2

SCORE_FILE=$(mktemp xtrack_score.XXX.csv)

python dstc_scripts/score.py \
    --dataset ${DATASET} \
    --dataroot ${DATA_DIRECTORY}/dstc2/data \
    --trackfile ${TRACK_FILE} \
    --ontology dstc_scripts/config/ontology_dstc2.json \
    --scorefile ${SCORE_FILE}

python dstc_scripts/report.py --scorefile ${SCORE_FILE}
#rm $SCORE_FILE
echo $SCORE_FILE
