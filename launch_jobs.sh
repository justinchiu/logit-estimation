#!/bin/bash

for i in {0..5}
do
    for N in 1000 2000 4000 8000 16000 32000
    do
        sbatch launch.sub "$i" "$N" 1e-8
    done
done
