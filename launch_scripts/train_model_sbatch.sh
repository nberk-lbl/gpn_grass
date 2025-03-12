#!/bin/bash
#SBATCH --job-name=4train_model
#SBATCH --output=train_model4.log
#SBATCH --error=train_model4.err
#SBATCH --constraint=gpu
#SBATCH --account=m342
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=4
#SBATCH --qos=regular

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh 
conda activate gpn 

WANDB_PROJECT=gpn_grass torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm \
    --do_train --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name nberkowitz/mini_grass \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 8 \
    --seed 42 \
    --save_strategy steps \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --logging_steps 10000 \
    --max_steps 120000 \
    --warmup_steps 1000 \
    --learning_rate 1e-3 \
    --lr_scheduler_type constant_with_warmup \
    --run_name train_gpn_grass \
    --output_dir /pscratch/sd/n/nberk/results/model2/ \
    --cache_dir /tmp \
    --model_type GPN \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 4 \
    --total_batch_size 1024 \
    --ddp_find_unused_parameters False \
    --bf16 --bf16_full_eval