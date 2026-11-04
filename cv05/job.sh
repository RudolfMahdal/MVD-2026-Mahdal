#!/bin/bash
#PBS -N encode_wiki_gpu
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=00:30:00
#PBS -q gpu
#PBS -j oe

cd $PBS_O_WORKDIR

module add mambaforge
export HF_HOME=$SCRATCHDIR
export HF_DATASETS_CACHE=$SCRATCHDIR

echo "Spouštím vektorizaci Wikipedie na GPU..."
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python encode_wiki.py