export CUDA_VISIBLE_DEVICES=""
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass python -m pdb -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to none \
    --dataset_name workflow/make_dataset/workflow/results/dataset \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir ./results/model \
    --model_type GPN \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 4 \
    --max_steps 10 \
    --logging_steps 1 \
    --save_steps 10 \
    --eval_steps 10 \
    --warmup_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant_with_warmup \
    --run_name gpn_grass_laptop
