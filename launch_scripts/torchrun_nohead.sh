module load conda
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass torchrun --nproc_per_node=4 -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name workflow/make_dataset/workflow/results/dataset \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 16 \
    --seed 42 \
    --save_strategy steps \
    --save_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --max_steps 120 \
    --warmup_steps 10 \
    --learning_rate 1e-3 \
    --lr_scheduler_type constant_with_warmup \
    --run_name gpn_grass \
    --output_dir workflow/make_dataset/workflow/results/model \
    --model_type GPN \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --total_batch_size 2048 \
    --torch_compile \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval
# conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm --do_train --do_eval \
#     --report_to wandb --prediction_loss_only True --remove_unused_columns False \
#     --dataset_name workflow/make_dataset/workflow/results/dataset --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
#     --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
#     --weight_decay 0.01 --optim adamw_torch \
#     --dataloader_num_workers 16 --seed 10 \
#     --save_strategy steps --save_steps 1000 --evaluation_strategy steps \
#     --eval_steps 1000 --logging_steps 1000 --max_steps 5000 --warmup_steps 200 \
#     --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
#     --run_name gpn_grass --output_dir workflow/make_dataset/workflow/results/model --model_type GPN \
#     --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1 \
#     --torch_compile \
#     --ddp_find_unused_parameters False \
#     --bf16 --bf16_full_eval


# salloc --nodes 1 --qos interactive --time 01:00:00 --account=m342 -C gpu -J tiberius_gpu

#salloc --nodes 1 --qos interactive --time 24:00:00 --account=m342 -C gpu --gpus 4 -J grass_minimal
#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --account=m342
#SBATCH -J tiberius_gpu
#SBATCH --mail-user=tbruna@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00


# srun --error=slurm_error_srun.log  --output=slurm_output_srun.log conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm --do_train --do_eval \
#     --report_to wandb --prediction_loss_only True --remove_unused_columns False \
#     --dataset_name workflow/make_dataset/workflow/results/dataset --tokenizer_name nberkowitz/tokenizer-dna-mlm \
#     --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
#     --weight_decay 0.01 --optim adamw_torch \
#     --dataloader_num_workers 16 --seed 42 \
#     --save_strategy steps --save_steps 1 --evaluation_strategy steps \
#     --eval_steps 2 --logging_steps 2 --max_steps 4 --warmup_steps 2 \
#     --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
#     --run_name gpn_grass --output_dir /pscratch/sd/n/nberk/results/model --model_type GPN \
#     --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
#     --torch_compile \
#     --ddp_find_unused_parameters False \
#     --bf16 --bf16_full_eval
