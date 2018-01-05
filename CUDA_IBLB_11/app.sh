#! /bin/bash -l

module load libs/CUDA

module load dev/gcc/4.9.4

./app 6 2 1.0 100000 10000 100
