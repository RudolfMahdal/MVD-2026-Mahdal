#!/bin/bash
#PBS -N tsne_mnist
#PBS -l select=1:ncpus=2:mem=50gb:scratch_local=10gb
#PBS -l walltime=12:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# uprava prostredi
module add mambaforge

/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python 3.py
