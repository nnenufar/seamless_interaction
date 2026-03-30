#!/bin/bash
#SBATCH --job-name=SI_preprocess
#SBATCH --output=./slurm/out/preprocess.out
#SBATCH --error=./slurm/out/preprocess.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 20
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

source ~/miniconda3/bin/activate
conda activate SI
python ./scripts/preprocess.py \
       --input_dir /hadatasets/joao.lima/data/seamless_interaction/naturalistic/ \
       --output_dir /hadatasets/joao.lima/data/seamless_interaction/naturalistic/processed\
       --num_workers 20