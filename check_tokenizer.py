from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/tokenizer-dna-mlm")

# Tokenize a DNA sequence
sequence = "ATCGATCG"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # Example output: ['A', 'T', 'C', 'G', 'A', 'T', 'C', 'G']

# Convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)  # Example output: [1, 2, 3, 4, 1, 2, 3, 4]
