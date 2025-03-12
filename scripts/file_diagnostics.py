#!/usr/bin/env python
# strict_diagnostics.py

import os
import time
from datasets import load_dataset

# Hardcoded single file test
TEST_FILE = "/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/shard_00001.jsonl"

def strict_test():
    print("=== Strict Diagnostic Test ===")
    
    # 1. Verify direct access
    print("1. Direct file access check...")
    try:
        with open(TEST_FILE, "r") as f:
            line = f.readline()
            print(f"Success: Read {len(line)} characters")
    except Exception as e:
        print(f"Failed: {str(e)}")
        return

    # 2. HF Dataset direct load
    print("\n2. Hugging Face Dataset load...")
    try:
        start_time = time.time()
        dataset = load_dataset("json", data_files={"train": [TEST_FILE]}, streaming=True)["train"]
        print(f"Dataset initialized in {time.time()-start_time:.1f}s")
        
        # 3. First sample access
        print("\n3. First sample access...")
        start_time = time.time()
        sample = next(iter(dataset))
        print(f"Sample retrieved in {time.time()-start_time:.1f}s")
        print(f"Keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"Failed: {str(e)}")

if __name__ == "__main__":
    strict_test()