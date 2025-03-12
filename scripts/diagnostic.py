import json
import os
import glob
from torch.utils.data import Dataset, DataLoader

class JsonlDataset(Dataset):
    def __init__(self, file_pattern):
        self.files = sorted(glob.glob(file_pattern))
        self.samples = []
        
        print(f"Found {len(self.files)} files matching pattern")
        
        # Validate file content
        for idx, f in enumerate(self.files):
            print(f"Processing file {idx+1}/{len(self.files)}: {os.path.basename(f)}")
            try:
                with open(f, "r") as infile:
                    file_lines = list(infile)
                    print(f"  Lines in file: {len(file_lines)}")
                    
                    for line_num, line in enumerate(file_lines):
                        try:
                            data = json.loads(line)
                            if "text" not in data:  # Replace with your key
                                print(f"  Line {line_num+1}: Missing 'text' key")
                            self.samples.append(data)
                        except json.JSONDecodeError:
                            print(f"  Line {line_num+1}: Invalid JSON")
            except Exception as e:
                print(f"Failed to read {f}: {str(e)}")
                raise
                
        print(f"Total samples loaded: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]["text"]  # Verify this key matches your data

# Test Execution
dataset = JsonlDataset("/pscratch/sd/n/nberk/results/small_dataset_fixed_uncompressed/train/shard_*.jsonl")
dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

if len(dataset) > 0:
    print(next(iter(dataloader)))
else:
    print("Dataset is empty - check input files and JSON formatting")