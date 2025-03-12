import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import sys

n_neighbors = 20
min_dist = 0.2
run_cfg = json.loads(open(sys.argv[1], 'r').read())
pfx = run_cfg['out_pfx']
window_file = run_cfg['out_pfx']+ ".windows_intergenic_repeat_padded.csv"

X = np.loadtxt(f'{run_cfg["output_dir"]}/{pfx}.avg_embeddings_N.tsv', delimiter='\t', skiprows=1)
labels = []

with open(f'data/{window_file}') as label_file:
    i = 0
    for n in label_file:
        chrom, bin, label = n.rstrip().split(",")
        labels.append(label)  
    
# filter out ambiguous
labels_array = np.array(labels)
mask = (labels_array != "ambiguous")

X_filtered = X[mask]
labels_filtered = labels_array[mask]
labels_filtered = labels_filtered.tolist()


def run_pipeline(pipeline, d, nei, p):
    # Fit and transform the data
    X_umap = pipeline.fit_transform(X_filtered)
    np.savetxt(f'data/TAIR10.allchrom_{d}_{nei}_{p}.UMAP.csv', X_umap, delimiter=',')

    plt.figure(figsize=(8, 6))

    # Convert data into a dataframe for seaborn
    df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
    df["label"] = labels_filtered 
    df.to_csv('TAIR10_umap_labeled.csv', index=False)

    g = sns.scatterplot(
        data=df, 
        x="UMAP1", 
        y="UMAP2", 
        hue="label", 
        palette="tab10",
        s=0.1)

    plt.legend(markerscale=40.0)
    plt.title("Genomic Bins UMAP")
    plt.savefig(f"data/AT_umap.png")
    plt.close()


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('umap', umap.UMAP(
        random_state=10,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    ))
])
run_pipeline(pipe, min_dist, n_neighbors, 0)
