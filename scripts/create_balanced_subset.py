import pandas as pd
from sklearn.utils import shuffle

# Paths to the input files
embeddings_file = '/pscratch/sd/n/nberk/grass/embeddings.tsv'
labels_file = '/pscratch/sd/n/nberk/grass/labels_subset.tsv'

# Read embeddings and labels with Pandas
embeddings_df = pd.read_csv(embeddings_file, sep='\t', header=None)
labels_df = pd.read_csv(labels_file, sep='\t', usecols=[2], names=['label'], header=None)

# Combine embeddings and labels into a single DataFrame
combined_df = pd.concat([embeddings_df, labels_df], axis=1)

# Shuffle the combined DataFrame
combined_df = shuffle(combined_df, random_state=10)

# Balance the categories
balanced_df = combined_df.groupby('label').head(50000)  # Adjust the number as needed

# Shuffle again to ensure the order is scrambled
balanced_df = shuffle(balanced_df, random_state=10)

# Reset the index
balanced_df.reset_index(drop=True, inplace=True)

# Save the balanced and shuffled DataFrame to a new file
balanced_df.to_csv('/pscratch/sd/n/nberk/grass/balanced_shuffled_dataset.tsv', sep='\t', index=False, header=False)

# Extract the labels and save to a separate file
balanced_labels_df = balanced_df[['label']]
balanced_labels_df.to_csv('/pscratch/sd/n/nberk/grass/balanced_shuffled_labels.tsv', sep='\t', index=False, header=False)

print("Balanced and shuffled dataset has been saved to 'balanced_shuffled_dataset.tsv'.")
print("Balanced and shuffled labels have been saved to 'balanced_shuffled_labels.tsv'.")