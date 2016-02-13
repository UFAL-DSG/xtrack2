#!/bin/bash
set -x
WHOLE_E_NAME="asruJ$2"
DATA_BUILD_FLAGS="--train_nbest_entries 0,1 --tagged"
XTRACK_FLAGS="--lr 0.001 --opt_type adam --mb_size 10 --n_cells 400 --emb_size 200 --input_activation rectify --input_n_hidden 300 --rnn_n_layers 1 --wcn_aggreg max --n_early_stopping 1 --ftr_emb_size 50 --input_n_layers 2 --x_include_score --oclf_n_layers 1 --l1 1.0 --n_epochs 20 --x_include_orig"

. ./run_gpu_exp_joint.sh
