#!/usr/bin/env python
# data_diagnostics.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def test_pipeline_interactive():
    """Interactive mode-specific data pipeline test"""
    print("=== Starting interactive diagnostic test ===")
    
    # 1. CUDA Availability Check
    print("1/5 Checking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 2. Dataset Access Check
    print("\n2/5 Checking dataset access...")
    data_path = "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/*.jsonl"
    try:
        dataset = load_dataset("json", data_files={"train": data_path}, streaming=True)["train"]
        first_sample = next(iter(dataset))
        print("Dataset access successful!")
        print(f"First sample keys: {list(first_sample.keys())}")
    except Exception as e:
        print(f"Dataset access failed: {str(e)}")
        return

    # 3. Tokenizer Initialization
    print("\n3/5 Initializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Tokenizer loading failed: {str(e)}")
        return

    # 4. Single Sample Processing
    print("\n4/5 Processing sample...")
    try:
        tokenized = tokenizer(
            first_sample["text"],
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        print(f"Tokenization successful! Shape: {tokenized['input_ids'].shape}")
    except Exception as e:
        print(f"Tokenization failed: {str(e)}")
        return

    # 5. GPU Transfer Test
    print("\n5/5 Testing GPU transfer...")
    try:
        if torch.cuda.is_available():
            gpu_tensor = tokenized["input_ids"].cuda()
            print(f"GPU tensor device: {gpu_tensor.device}")
            print(f"GPU tensor value: {gpu_tensor[0, :5]} (first 5 tokens)")
        else:
            print("No CUDA available, skipping GPU transfer")
    except Exception as e:
        print(f"GPU transfer failed: {str(e)}")
        return

    print("\n=== All checks passed! ===")

if __name__ == "__main__":
    test_pipeline_interactive()