from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load resources
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gonzalobenegas/tokenizer-dna-mlm')
    
    print("Loading dataset...")
    dataset = load_dataset('nberkowitz/mini_grass', split='train[:3]')  # First 3 samples
except Exception as e:
    print(f"LOAD ERROR: {str(e)}")
    exit(1)

# 2. Validate sequence casing
for i in range(3):
    seq = dataset[i]['seq']
    print(f"\nSample {i}:")
    print(f"Original: {seq[:50]}... (length: {len(seq)})")
    print(f"Lowercase: {seq.lower()[:50]}...")
    
    # 3. Test tokenization
    try:
        tokens = tokenizer(seq.lower(), truncation=True, max_length=512)
        print(f"Tokenized length: {len(tokens.input_ids)}")
        print(f"Decoded start: {tokenizer.decode(tokens.input_ids[:5])}")
    except Exception as e:
        print(f"TOKENIZATION ERROR: {str(e)}")

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device 0 name: {torch.cuda.get_device_name(0)}")
    t = torch.randn(3,3).cuda()
    print(f"Tensor on GPU: {t.device}")