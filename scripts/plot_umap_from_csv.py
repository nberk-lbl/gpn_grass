#import numpy as np
#from sklearn.decomposition import PCA
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import sys

df = pd.read_csv('umap_tair_sorted.csv')

# Create the plot
plt.figure(figsize=(8, 6))

# Get the unique labels
labels = df['label'].unique()

# Create a palette and modify the color of the third label (index 2) to cyan
palette = sns.color_palette("tab10", n_colors=len(labels))
palette[2] = 'cyan'

# Create the scatter plot with the modified palette
g = sns.scatterplot(
    data=df, 
    x="UMAP1", 
    y="UMAP2", 
    hue="label", 
    palette=palette,
    s=0.1)

# Update the legend
handles, labels = g.get_legend_handles_labels()
plt.legend(handles, labels, markerscale=40.0)

# Add title and save the plot
plt.title("Genomic Bins UMAP")
plt.savefig(f"AT_umap_3.png")
plt.close()

"""
plt.figure(figsize=(8, 6))

plt.legend(markerscale=40.0)
plt.title("Genomic Bins UMAP")
plt.savefig(f"data/AT_umap_1.png")
plt.close()
"""

