import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Path to the confusion matrix CSV file
conf_matrix_file = 'confusion_matrix.csv'

# Read the confusion matrix from the CSV file
conf_matrix_df = pd.read_csv(conf_matrix_file, index_col=0)

# Convert the DataFrame to a numpy array
conf_matrix = conf_matrix_df.values

# Extract the labels from the DataFrame index
labels = conf_matrix_df.index

# Create the confusion matrix plot
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figure size for long labels

# Use seaborn heatmap to plot the confusion matrix without values
sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)

# Increase font size for labels
ax.set_xticklabels(labels, rotation=90, fontsize=14, ha='right')
ax.set_yticklabels(labels, fontsize=14)

# Set titles and labels
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)

# Adjust margins to create more space for the labels
plt.subplots_adjust(left=0.6, bottom=0.6)  # Increase these values as needed

plt.savefig('confusion_matrix.png')
plt.close()