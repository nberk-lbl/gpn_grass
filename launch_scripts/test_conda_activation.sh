#!/bin/sh
#SBATCH --job-name=test_conda
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C=gpu
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH --time=00:10:00
#SBATCH --output=test_conda_output.log
#SBATCH --error=test_conda_error.log
module load conda
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn python -c "import torch; print(torch.__version__)"
