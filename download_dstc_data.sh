#!/bin/bash
set -e

# Go to the directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
cd $DIR

# Load configuration.
. ./config.sh

# Create the destination directory.
mkdir -p ${DATA_DIRECTORY}

# Download & unpack DSTC2 data.
DSTC2_DIR=${DATA_DIRECTORY}/dstc2
mkdir -p ${DSTC2_DIR}
wget --continue http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz \
                -O ${DSTC2_DIR}/dstc2_traindev.tar.gz
wget --continue http://camdial.org/~mh521/dstc/downloads/dstc2_test.tar.gz \
                -O ${DSTC2_DIR}/dstc2_test.tar.gz

cd ${DSTC2_DIR}
tar xzvf dstc2_traindev.tar.gz
tar xzvf dstc2_test.tar.gz

