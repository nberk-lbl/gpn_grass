salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account=m342  --mem=64G --gres=gpu:4

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account=m342  --mem=64G --gres=gpu:4 --gpu-bind=closest 


srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
    --gpu-bind=closest,single:1 \
    --cpus-per-task=8 \
    --export=ALL \
    python test_distributed_dataset.py

python -m gpn.ss.run_mlm \
  --dataset_name $TEST_DIR \
  --model_type GPN \
  --output_dir $TEST_DIR/output \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --max_steps 2 \
  --save_steps 1 \
  --logging_steps 1 \
  --overwrite_output_dir



  "/pscratch/sd/n/nberk"

python -c 'import json; seq = "ACGT" * 100; open(f"/pscratch/sd/n/nberk/train/shard1.jsonl", "w").write(json.dumps({"text": seq}) + "\n")'
python -c 'import json; seq = "TGCA" * 100; open(f"/pscratch/sd/n/nberk/validation/shard1.jsonl", "w").write(json.dumps({"text": seq}) + "\n")'
