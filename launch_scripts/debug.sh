#!/bin/sh
#SBATCH --job-name=gpn_grass
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=debug2_output_s.log
#SBATCH --error=debug2_error_s.log
module load conda
export TORCH_DISTRIBUTED_DEBUG=DETAIL
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass python -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to none \
    --dataset_name /pscratch/sd/n/nberk/results/dataset \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir /pscratch/sd/n/nberk/results/model \
    --model_type GPN \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --max_steps 10 \
    --logging_steps 1 \
    --save_steps 10 \
    --eval_steps 10 \
    --warmup_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant_with_warmup \
    --run_name gpn_grass_debug
