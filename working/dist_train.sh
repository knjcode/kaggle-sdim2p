#!/bin/bash

# bash dist_train.sh <num_gpus> <conf_file> <options>
# bash dist_train.sh 8 conf/test.yaml <options>

worker_num=$1
conf_file=$2
options=${@:3:($#-2)}

OMP_NUM_THREADS="1" \
torchrun --standalone --nnodes=1 --nproc_per_node=$1 \
dist_train.py -c ${conf_file} ${options}
