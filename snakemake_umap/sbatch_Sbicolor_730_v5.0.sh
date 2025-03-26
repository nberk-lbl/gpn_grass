#!/bin/sh
#SBATCH --job-name=Sbicolor_730_v5.0
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=01:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output=Sbicolor_730_v5.0.log
#SBATCH --error=Sbicolor_730_v5.0.err

module load conda
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh
conda activate gpn

srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Sbicolor_730_v5.0.json
