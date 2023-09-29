#!/bin/bash

for i in {1..5}
do
    sbatch launch.sub $i 32000 1e-3
    sbatch launch.sub $i 32000 1e-2
    sbatch launch.sub $i 32000 1e-1
done
