#!/bin/bash
WHOLE_E_NAME="asruX3_lmemb$2"
DATA_BUILD_FLAGS="--train_nbest_entries 0,1 --tagged"
XTRACK_FLAGS="--lr 0.001 --opt_type adam --mb_size 10 --n_cells 300 --emb_size 170 --input_activation rectify --input_n_hidden 300 --rnn_n_layers 1 --wcn_aggreg max --n_early_stopping 1 --input_n_layers 2 --x_include_score --oclf_n_layers 0 --load_params /tmp/pretrain.params"

. ./run_gpu_exp.sh