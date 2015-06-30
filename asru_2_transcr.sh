#!/bin/bash
WHOLE_E_NAME="asru_2_transcr"
DATA_BUILD_FLAGS="--train_nbest_entries 0,1"
XTRACK_FLAGS="--lr 0.001 --opt_type adam --mb_size 10 --n_cells 100 --emb_size 170 --input_n_layers 1 --input_activation rectify --input_n_hidden 300 --rnn_n_layers 1 --wcn_aggreg max --n_early_stopping 1 --ftr_emb_size 50 --input_n_layers 1 --x_include_score"

. ./run_gpu_exp.sh 1
