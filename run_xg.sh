#!/bin/sh
#SBATCH --job-name=xg_boost
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH --mem=128G
#SBATCH -C=gpu
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=xg_boost3.log
#SBATCH --error=xg_boost3.err
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn python train_xgboost_on_embeddings.py
