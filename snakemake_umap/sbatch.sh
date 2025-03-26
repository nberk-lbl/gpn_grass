#!/bin/sh
#SBATCH --job-name=2run_gpn
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=2run_gpn.log
#SBATCH --error=2run_gpn.err

module load conda
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh
conda activate gpn

srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Bdistachyon_556_v3.0.json &
srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Pvirgatum_516_v5.0.json &
srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/OsativaKitaake_499_v3.0.json &
srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Zmays_833_Zm-B73-REFERENCE-NAM-5.0.json &
srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Sbicolor_730_v5.0.json &
srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/Sviridis_726_v4.0.json &
wait
