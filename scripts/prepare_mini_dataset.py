import os
import zstandard as zstd
from datasets import load_dataset

# Load original dataset (WITHOUT RENAMING HERE)
dataset = load_dataset(
    "json",
    data_dir="/pscratch/sd/n/nberk/results/dataset/data",
    data_files={"train": "train/shard_*.jsonl.zst", "validation":"validation/shard_*.jsonl.zst"},
    cache_dir="/pscratch/sd/n/nberk/datasets_cache",
)

# RENAME COLUMN FIRST
dataset = dataset.rename_column("seq", "text")

# Create subset from renamed dataset
mini_dataset = {
    "train": dataset["train"].shuffle().select(range(512)),  # Increase to 512
    "validation": dataset["validation"].shuffle().select(range(128))  # 128 validation
}


# Save logic (WITH EXPLICIT COLUMN CHECK)
output_dir = "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed"

os.makedirs(f"{output_dir}/train", exist_ok=True)
os.makedirs(f"{output_dir}/validation", exist_ok=True)

# Save TRAIN split without compression
mini_dataset["train"].select(range(100)).to_json(f"{output_dir}/train/shard_00001.jsonl")

# Save VALIDATION split without compression
mini_dataset["validation"].select(range(20)).to_json(f"{output_dir}/validation/shard_00001.jsonl")


