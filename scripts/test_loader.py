# test_loader.py
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
from datasets import load_dataset

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "Path to dataset directory containing train/validation subdirs"}
    )
    dataset_config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_auth_token: bool = field(default=False)
    max_samples: int = field(default=2, metadata={"help": "Number of samples to test"})

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def main():
    configure_logging()
    
    # Parse arguments correctly
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    
    logging.info(f"Loading dataset from: {data_args.dataset_name}")
    
    try:
        # Proper load_dataset call for directory structure
        raw_datasets = load_dataset(
            "json",
            data_dir=data_args.dataset_name,
            data_files={
                "train": "train/shard*.jsonl",
                "validation": "validation/shard*.jsonl"
            },
            cache_dir=data_args.cache_dir,
            streaming=True
        )
        
        logging.info("Dataset loaded successfully. Splits: %s", list(raw_datasets.keys()))
        
        # Test sample iteration
        for split in raw_datasets:
            logging.info(f"Testing {split} split:")
            for i, example in enumerate(raw_datasets[split]):
                if i >= data_args.max_samples:
                    break
                logging.info(f"Sample {i+1}: {example['text'][:50]}...")
                
        logging.info("All tests passed!")

    except Exception as e:
        logging.error(f"Dataset load failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

