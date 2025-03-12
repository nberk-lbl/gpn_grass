#!/bin/bash
# validate_tokenizer_with_conda.sh

# Set up Conda environment
CONDA_PATH="/global/homes/n/nberk/miniconda3"
ENV_NAME="gpn"

echo "================================================"
echo " Tokenizer Validation Test (with Conda)"
echo "================================================"

# Load Conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
echo -e "\nActivating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Verify Python path
echo -e "\nPython path: $(which python)"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"

# Run validation tests
echo -e "\nRunning tokenizer tests..."
echo "================================================"

# Test 1: Lowercase sequence
echo -e "\n[Test 1] Lowercase DNA sequence:"
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gonzalobenegas/tokenizer-dna-mlm')
seq = 'acgtacgt'
print(f'Original: {seq}')
print('Tokenized:', tok.encode(seq))
print('Decoded:', tok.decode(tok.encode(seq)))
"

# Test 2: Uppercase sequence
echo -e "\n[Test 2] Uppercase DNA sequence:"
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gonzalobenegas/tokenizer-dna-mlm') 
seq = 'ACGTACGT'
print(f'Original: {seq}')
print('Tokenized:', tok.encode(seq))
print('Decoded:', tok.decode(tok.encode(seq)))
"

# Test 3: Actual dataset sample
echo -e "\n[Test 3] Dataset sample processing:"
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer

print('\nLoading dataset...')
ds = load_dataset('nberkowitz/mini_grass', split='train[:1]')
tok = AutoTokenizer.from_pretrained('gonzalobenegas/tokenizer-dna-mlm')

print('\nRaw sample:', ds[0]['seq'])
print('Lowercased:', ds[0]['seq'].lower())

print('\nTokenizing raw:')
print(tok.encode(ds[0]['seq']))

print('\nTokenizing lowercased:')
print(tok.encode(ds[0]['seq'].lower()))
"

echo "================================================"
echo " Validation Complete"
echo "================================================"