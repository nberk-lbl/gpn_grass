
#!/bin/sh
#SBATCH --job-name=run_gpn
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=08:00:00
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/Bdistachyon_556_v3.0.json &
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/Sbicolor_730_v5.0.json &
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/Pvirgatum_516_v5.0.json &
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/Zmays_833_Zm-B73-REFERENCE-NAM-5.0.json &
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/OsativaKitaake_499_v3.0.json & 
srun python /global/homes/n/nberk/gpn_grass/scripts/sk_umap.py /pscratch/sd/n/nberk//intermediate_files/Sviridis_726_v4.0.json
