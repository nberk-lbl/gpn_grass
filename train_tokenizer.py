from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Set up a trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,  # Adjust based on your dataset size
    min_frequency=2,   # Minimum frequency for a token to be included
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train the tokenizer on your dataset
tokenizer.train(files=[
    "workflow/make_dataset/workflow/results/genome/Phallii_590_v3.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Bdistachyon_556_v3.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Sviridis_726_v4.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Sbicolor_730_v5.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Pvirgatum_516_v5.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Tintermedium_770_v3.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/Zmays_833_Zm-B73-REFERENCE-NAM-5.0.fa.gz",
    "workflow/make_dataset/workflow/results/genome/OsativaKitaake_499_v3.0.fa.gz"
], trainer=trainer)

# Save the tokenizer
tokenizer.save("nb_tokenizer.json")