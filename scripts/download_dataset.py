from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("nberkowitz/mini_grass_text", streaming=True)

# Iterate with progress bar
for batch in tqdm(dataset["train"].take(100)):
    print(batch)  # Verify 1 sample loads