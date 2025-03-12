#!/bin/bash
#SBATCH --job-name=train_tokenizer       # Name of the job
#SBATCH --time=08:00:00                   # Time limi
#SBATCH --output=train_tokenizer_%j.log   # Output log file
#SBATCH --error=train_tokenizer_%j.err    # Error log file
#SBATCH -C=gpu
#SBATCH --account=m342                    # Specify the account (if applicable)

# Load necessary modules
module load conda

# Activate your Conda environment
source activate /global/homes/n/nberk/miniconda3/envs/gpn

# Run the tokenizer training script
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn python train_tokenizer.py