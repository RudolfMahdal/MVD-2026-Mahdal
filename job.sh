#!/bin/bash
#PBS -N MVD_cv01_mahdal
#PBS -l select=1:ncpus=2:mem=60gb:scratch_local=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

echo "Spouštím výpočet přes Conda prostředí..."
/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python cv01.py

