#!/bin/bash

for i in {0..100}
do
    sbatch launch.sub $i "1e-6"
    sbatch launch.sub $i "1e-5"
    sbatch launch.sub $i "1e-4"
    sbatch launch.sub $i "1e-3"
    sbatch launch.sub $i "1e-2"
    sbatch launch.sub $i "1e-1"
    sbatch launch.sub $i "5e-1"
    sbatch launch.sub $i "1"
done
