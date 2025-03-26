
names = [
    "Bdistachyon_556_v3.0",
    "Pvirgatum_516_v5.0",
    "OsativaKitaake_499_v3.0",
    "Zmays_833_Zm-B73-REFERENCE-NAM-5.0",
    "Sbicolor_730_v5.0",
    "Sviridis_726_v4.0"
    ]

template = \
"""#!/bin/sh
#SBATCH --job-name={n}
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=01:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output={n}.log
#SBATCH --error={n}.err

module load conda
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh
conda activate gpn

srun --exclusive -n1 python /global/homes/n/nberk/gpn_grass/scripts/run_gpn.py /pscratch/sd/n/nberk/intermediate_files/{n}.json"""

for n in names:
    with open(f"sbatch_{n}.sh", 'w') as f:
        print(template.format(n=n), file=f)





