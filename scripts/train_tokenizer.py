from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, normalizers
from tokenizers.normalizers import Lowercase

ds = load_dataset("nberkowitz/gpn_grass", split="train")
sampled_seqs = ds.shuffle()["seq"] #.select(range(10_000))["seq"] 

# BPE is ideal for DNA
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence([Lowercase()])

trainer = trainers.BpeTrainer(
    vocab_size=10_000,  # Adjust based on k-mer diversity
    special_tokens=["[PAD]", "[MASK]", "[UNK]"],
    min_frequency=2
)

# Train on sequences
tokenizer.train_from_iterator(
    sampled_seqs,  # Or full dataset
    trainer=trainer,
    length=len(sampled_seqs)
)

# Save
tokenizer.save("gpn-grass-tokenizer-full.json")
