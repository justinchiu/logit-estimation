#!/bin/bash

for i in {1..5}
do
    sbatch launch.sub $i 32000 1e-8
    sbatch launch.sub $i 32000 1e-6
    sbatch launch.sub $i 32000 1e-4
    sbatch launch.sub $i 32000 1e-7
    sbatch launch.sub $i 32000 1e-5
    sbatch launch.sub $i 32000 1e-3
done
