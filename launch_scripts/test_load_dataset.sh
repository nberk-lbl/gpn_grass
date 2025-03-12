#!/bin/bash
#SBATCH --job-name=DATA_TEST
#SBATCH --output=load_data_test.log
#SBATCH --error=load_data_test.err
#SBATCH --constraint=gpu
#SBATCH --account=m342
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh
conda activate gpn
python -c "from datasets import load_dataset; d=load_dataset( \
        \"nberkowitz/gpn_grass\", \
        None, \
        cache_dir=\"/tmp\", \
        use_auth_token=None, \
        streaming=True\
    ); print(d)"