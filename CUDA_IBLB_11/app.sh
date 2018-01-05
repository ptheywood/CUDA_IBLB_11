#! /bin/bash -l

module load libs/CUDA/8.0.44/binary

module load dev/gcc/4.9.4

./app 6 1 1.0 100000 10000 100 1 0
