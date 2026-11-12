#!/bin/bash
#PBS -N train_som_wiki
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=02:00:00
#PBS -q gpu
#PBS -j oe

cd $PBS_O_WORKDIR

module add mambaforge
export HF_HOME=$SCRATCHDIR
export HF_DATASETS_CACHE=$SCRATCHDIR
export TORCHDYNAMO_DISABLE=1

echo "Starting SOM training (40x40, 3 epochs) on Wikipedia embeddings..."
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python train_som.py