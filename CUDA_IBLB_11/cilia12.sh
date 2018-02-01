#! /bin/bash -l

module load libs/CUDA/8.0.44/binary

module load dev/gcc/4.9.4

./app 1 12 1.0 100000 100000 100 1 0

./app 5 12 1.0 100000 100000 100 1 0

./app 7 12 1.0 100000 100000 100 1 0

./app 11 12 1.0 100000 100000 100 1 0


