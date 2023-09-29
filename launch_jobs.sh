#!/bin/bash

for i in {0..100}
do
    for k in {1000 2000 4000 8000 16000 32000}
    do
        sbatch launch.sub i
    done
done
