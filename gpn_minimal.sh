#!/bin/sh
python -m pdb -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name /pscratch/sd/n/nberk/results//dataset \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 16 \
    --seed 10 \
    --save_strategy steps \
    --save_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --max_steps 120 \
    --warmup_steps 10 \
    --learning_rate 1e-1 \
    --lr_scheduler_type constant_with_warmup \
    --run_name gpn_grass \
    --output_dir /pscratch/sd/n/nberk/results/model \
    --model_type GPN \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --total_batch_size 32 \
    --torch_compile \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval

