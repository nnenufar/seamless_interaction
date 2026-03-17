#!/bin/bash
#SBATCH --job-name=SI_preprocess
#SBATCH --output=./slurm/out/preprocess.out
#SBATCH --error=./slurm/out/preprocess.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

source ~/miniconda3/bin/activate
conda activate SI
python ./scripts/preprocess.py \
       --input_dir ./inspect/samples \
       --output_dir ./inspect/processed\
       --num_workers 4