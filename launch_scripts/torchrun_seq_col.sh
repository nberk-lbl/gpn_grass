srun --gres=gpu:4 -N1 -t 02:00:00 bash -c '
source /global/homes/n/nberk/miniconda3/etc/profile.d/conda.sh &&
conda activate gpn &&
export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
export TOKENIZERS_PARALLELISM=false &&
torchrun --nproc_per_node=4 -m gpn.ss.run_mlm \
  --do_train \
  --do_eval \
  --dataset_name nberkowitz/mini_grass \
  --text_column_name seq \
  --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
  --output_dir /pscratch/sd/n/nberk/results/model/ \
  --per_device_train_batch_size 128 \
  --max_steps 100 \
  --eval_steps 20 \
  --logging_steps 10 \
  --save_steps 50 \
  --fp16 \
  --report_to none \
  --log_level debug \
  --ddp_timeout 1800
'
