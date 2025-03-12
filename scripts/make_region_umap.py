import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

n_neighbors = 20
min_dist = 0.2

labels_file = "TAIR10_chr_3,17800288,17833759_labels.tsv"
embeddings_file = "TAIR10_chr_3,17800288,17833759.tsv"

X = np.loadtxt(embeddings_file, delimiter='\t', skiprows=1)
region_labels = []
name_labels = []

with open(f'{labels_file}') as lbf:
    for n in lbf:
        idx, pos, region_label, name_label = n.split("\t")
        region_labels.append(region_label) 
        name_labels.append(name_label.rstrip()) 
    
# filter out ambiguous
labels_array = np.array(region_labels)

def run_pipeline(pipeline, labels_array, X, file_prefix):
    # Fit and transform the data
    X_umap = pipeline.fit_transform(X)

    plt.figure(figsize=(8, 6))

    # Convert data into a dataframe for seaborn
    df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
    df["label"] = labels_array 

    g = sns.scatterplot(
        data=df, 
        x="UMAP1", 
        y="UMAP2", 
        hue="label", 
        palette="tab10",
        s=0.1)

    plt.legend(markerscale=40.0)
    plt.title("file_prefix")
    plt.savefig(f"{file_prefix}.png")
    plt.close()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('umap', umap.UMAP(
        random_state=10,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    ))
])

run_pipeline(pipe, labels_array, X, labels_file[0:-4])
