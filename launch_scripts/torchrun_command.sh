#!/bin/sh
#SBATCH --job-name=gpn_grass
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --qos=regular
#SBATCH --account=m342
#SBATCH -C=gpu
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=tr_output.log
#SBATCH --error=tr_error.log
export TORCH_DISTRIBUTED_DEBUG=DETAIL
module load conda
conda run -p /global/homes/n/nberk/miniconda3/envs/gpn WANDB_PROJECT=gpn_grass torchrun --nproc_per_node=4 -m pdb -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name nberkowitz/mini_grass \
    --dataset_config_name "seq_as_text_col" \  # Your custom config ID
    --override_dataset_columns '{"text": "seq"}'  # Map seq → text
    --dataset_name nberkowitz/mini_grass \
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
    --output_dir /pscratch/sd/n/nberk/results/model \
    --model_type GPN \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --total_batch_size 2048 \
    --torch_compile \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval

salloc --nodes=1 --gres=gpu:4 --time=00:30:00 --partition=debug  # Request 30min interactive

# Once allocated, run:
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh &&
conda activate gpn &&
export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
export TOKENIZERS_PARALLELISM=false &&

torchrun --nproc_per_node=4 -m gpn.ss.run_mlm \
  --do_train \
  --max_steps 5 \
  --max_train_samples 100 \
  --dataset_name /pscratch/sd/n/nberk/datasets/mini_grass_text \
  --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
  --output_dir /pscratch/sd/n/nberk/results/model_test/ \
  --per_device_train_batch_size 64 \
  --logging_steps 1 \
  --save_steps 50 \
  --fp16 \
  --report_to none \
  --ddp_timeout 1800 \
  --disable_tqdm True


python -c "
from datasets import load_dataset
from transformers import AutoTokenizer

# Download dataset
dataset = load_dataset('nberkowitz/mini_grass_text')
dataset.save_to_disk('/pscratch/sd/n/nberk/datasets/mini_grass_text')
"


OK, the curl command returned

(gpn) nberk@nid200372:~/gpn_grass> curl -I "https://huggingface.co/datasets/nberkowitz/mini_grass_text"

HTTP/2 200

content-type: text/html; charset=utf-8
content-length: 411855
date: Sat, 08 Mar 2025 23:47:53 GMT
x-powered-by: huggingface-moon
cross-origin-opener-policy: same-origin
referrer-policy: strict-origin-when-cross-origin
x-request-id: Root=1-67ccd729-78d670dd0f38f7a5062e582a
x-frame-options: DENY
etag: W/"648cf-pzj4741r68Ovqfqc9l93A+YIz9M"
x-cache: Miss from cloudfront
via: 1.1 eb2d3c7af7f453242c6551f237e2edd4.cloudfront.net (CloudFront)
x-amz-cf-pop: MCI50-P1
x-amz-cf-id: qgmc2_4rbKcAy9Kl1npV2BtVlsjAH5gXA9eRI6Js-LAIISVmE4COzg==

According to the web interface the dataset is pubic. I get a similar result for mini_grass

the output seems writable
ls -ld /pscratch/sd/n/nberk/datasets/ 
drwxrwx--- 2 nberk nberk 4096 Mar  8 15:44 /pscratch/sd/n/nberk/datasets/

this times out:

timeout 5m python download_dataset

this has been running for a few minutes without output:

dataset = load_dataset("nberkowitz/mini_grass_text", streaming=True)

----------

srun --gres=gpu:4 -N1 -t 01:00:00 bash -c '
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh &&
conda activate gpn &&
WANDB_PROJECT=mini_test torchrun --nproc_per_node=1 -m gpn.ss.run_mlm
--do_train --do_eval --report_to none --dataset_name nberkowitz/mini_grass_text --tokenizer_name gonzalobenegas/tokenizer-dna-mlm --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 --weight_decay 0.01 --optim adamw_torch --dataloader_num_workers 0 --seed 42 --save_strategy no --evaluation_strategy no --max_steps 50 --warmup_steps 10 --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup --run_name mini_test --output_dir ./out --model_type GPN --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 1 \
'


srun ... torchrun ... -m gpn.ss.run_mlm \
    --dataset_name "." \
    --data_files_train "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/*.jsonl" \
    --data_files_validation "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/validation/*.jsonl" \


srun --nodes=1 --ntasks-per-node=1 \
     --gpus-per-task=4 \
     --cpus-per-task=8 \
     --time=01:00:00 \
     --partition=interactive \
     --account=m342 \
     --export=ALL,MODEL_TYPE=GPN \
     --pty /bin/bash -c '
     source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh && 
     conda activate gpn &&
     WANDB_PROJECT=gpn_debug \
     TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
     torchrun --nproc_per_node=4 -m gpn.ss.run_mlm \
         --do_train --do_eval \
            --dataset_name "." \
        --data_files_train "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/*.jsonl" \
        --data_files_validation "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/validation/*.jsonl" \
         --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
         --model_type "$MODEL_TYPE" \  # Use environment variable
         --per_device_train_batch_size 64 \
         --per_device_eval_batch_size 64 \
         --gradient_accumulation_steps 1 \
         --total_batch_size 256 \
         --max_steps 50 \
         --eval_steps 10 \
         --warmup_steps 5 \
         --learning_rate 1e-4 \
         --output_dir /pscratch/sd/n/nberk/results/mini_run_v4 \
         --run_name interactive_debug_v2 \
         --report_to wandb \
         --ddp_find_unused_parameters False \
         --bf16 \
         --bf16_full_eval \
         --seed 42'                                                                                                                                                        
         --do_train --do_eval \                                                                                                                                                                                  
         --dataset_name "." \       
         --data_files_train "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/*.jsonl" \
         --data_files_validation "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/validation/*.jsonl" \                                                                                                                   
         --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \                                                                                                                                                     
         --model_type GPN \                                                                                                                                                                                      
         --per_device_train_batch_size 64 \                                                                                                                                                                      
         --per_device_eval_batch_size 64 \                                                                                                                                                                       
         --gradient_accumulation_steps 1 \                                                                                                                                                                       
         --total_batch_size 256 \                                                                                                                                                                                
         --max_steps 50 \                                                                                                                                                                                        
         --eval_steps 10 \                                                                                                                                                                                       
         --warmup_steps 5 \                                                                                                                                                                                      
         --learning_rate 1e-4 \                                                                                                                                                                                  
         --output_dir /pscratch/sd/n/nberk/results/mini_run_v4 \                                                                                                                                                 
         --run_name interactive_debug_v2 \                                                                                                                                                                       
         --report_to wandb \                                                                                                                                                                                     
         --ddp_find_unused_parameters False \                                                                                                                                                                    
         --bf16 \                                                                                                                                                                                                
         --bf16_full_eval \                                                                                                                                                                                      
         --seed 42  

srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
    --gpu-bind=closest \
    --export=ALL \
    bash -c 'echo "Rank $SLURM_PROCID sees GPU $CUDA_VISIBLE_DEVICES" && nvidia-smi -L'


srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
    --gpu-bind=map_gpu:0,1,2,3 \
    --export=ALL \
    python test_distributed_dataset.py
    