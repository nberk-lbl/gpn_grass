#!/bin/sh
#SBATCH --job-name=tokenizer
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH --mem=32G
#SBATCH -C=gpu
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8 
#SBATCH --time=16:00:00
#SBATCH --export=ALL
#SBATCH --output=tokenizer.log
#SBATCH --error=tokenizer.err
module load conda
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn python train_tokenizer.py
