#!/bin/bash

for i in {0..100}
do
    for N in {1000 2000 4000 8000 16000 32000}
    do
        sbatch launch.sub i N
    done
done
