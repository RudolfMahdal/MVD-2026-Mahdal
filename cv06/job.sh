#!/bin/bash
#PBS -N extract_html_lm
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=01:00:00
#PBS -q gpu
#PBS -j oe

cd $PBS_O_WORKDIR

module add mambaforge
export HF_HOME=$SCRATCHDIR
export HF_DATASETS_CACHE=$SCRATCHDIR
export TORCHDYNAMO_DISABLE=1
if [ ! -f "curlie_cs.jsonl.gz" ]; then
    echo "Downloading curlie_cs.jsonl.gz..."
    wget --no-check-certificate https://ufal.mff.cuni.cz/~madr/curlie_cs.jsonl.gz
fi

echo "Starting HTML-LM vector extraction for raw and cleaned HTML..."
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python extract_vectors.py