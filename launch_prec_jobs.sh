#!/bin/bash

for i in {1..100}
do
    sbatch launch.sub $i "1e-6"
    sbatch launch.sub $i "1e-4"
    sbatch launch.sub $i "1e-3"
    sbatch launch.sub $i "1e-2"
    sbatch launch.sub $i "1e-1"
done
