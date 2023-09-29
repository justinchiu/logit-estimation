#!/bin/bash

for i in {0..50}
do
    for eps in 1e-6 1e-5 1e-4 1e-3 1e-2
    do
        sbatch launch.sub $i 32000 $eps
    done
done
