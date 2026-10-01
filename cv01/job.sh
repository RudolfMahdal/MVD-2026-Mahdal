#!/bin/bash
#PBS -N MVD_cv01_mahdal
#PBS -l select=1:ncpus=2:mem=60gb:scratch_local=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Přejdeme do složky, ze které byla úloha odeslána
cd $PBS_O_WORKDIR

echo "Starting..."
# Přímé zavolání vašeho Pythonu s NumPy ze složky envs
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python cv01.py

