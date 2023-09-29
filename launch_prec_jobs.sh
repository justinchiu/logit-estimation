#!/bin/bash

for i in {0..5}
do
    sbatch launch.sub $i 32000 1e-8
    sbatch launch.sub $i 32000 1e-6
    sbatch launch.sub $i 32000 1e-4
    sbatch launch.sub $i 32000 1e-2
done
