#!/bin/bash
#SBATCH --job-name=gpn-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4             # Request GPUs
#SBATCH --time=01:00:00
#SBATCH --output=job-%j.log

# Load Conda module (if required by your cluster)
module load conda || true

srun bash -c ""source activate /global/homes/n/nberk/miniconda3/envs/gpn &&

# Simplified command for debugging
WANDB_PROJECT=mini_grass torchrun \
    --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') \
    -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --dataset_name nberkowitz/mini_grass \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir /pscratch/sd/n/nberk/results/model/ \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --total_batch_size 512 \
    --dataloader_num_workers 4 \
    --max_steps 120 \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_steps 10 \
    --bf16 \
    --report_to none \
    --log_level debug"
