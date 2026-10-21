#!/bin/bash
#PBS -N build_wiki_index
#PBS -l select=1:ncpus=16:mem=128gb:scratch_local=10gb
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

module add mambaforge
export HF_HOME=$SCRATCHDIR
export HF_DATASETS_CACHE=$SCRATCHDIR
echo "Plny dataset 16 jader"
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python build_index.py --cores 16
