#!/bin/bash
#PBS -N train_w2v_gpu
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:mem=16gb
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

module add mambaforge

/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python train_word2vec.py
#/storage/brno2/home/rudolfmahdal/envs/rudolfmahdalmvd/bin/python train_word2vec.py cbow