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
  
#sample_size = 50000
n_neighbors = [5, 10, 20, 50, 100, 200]
min_dist = [0, .1, .2, .5, .8, 1]
#pca_comp = [16, 64, 128, 256]

run_cfg = json.loads(open(sys.argv[1], 'r').read())
pfx = run_cfg['out_pfx']
window_file = f'{run_cfg["output_dir"]}/{pfx}.windows_intergenic_repeat_padded.csv'

X = np.loadtxt(f'{run_cfg["output_dir"]}/{pfx}.avg_embeddings_N.tsv', delimiter='\t', skiprows=0)

labels = []

with open(window_file) as label_file:
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

    g = sns.scatterplot(
        data=df, 
        x="UMAP1", 
        y="UMAP2", 
        hue="label", 
        palette="tab10",
        s=2,
        alpha=0.2)

    plt.legend(title="Categories", markerscale=5.0)
    plt.title("Genomic Bins UMAP")
    plt.savefig(f"data/param_sweep/igpad/{pfx}_{d}_{nei}_igpad.png")
    plt.close()

for d in min_dist:
    for nei in n_neighbors:
        no_pca_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('umap', umap.UMAP(
                n_components=2, 
                random_state=10,
                n_neighbors=nei,
                min_dist=d
            ))
        ])
        run_pipeline(no_pca_pipe, d, nei, 0)
        """
        for p in pca_comp:
            print(f"{d}\t{nei}\t{p}")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=p)),
                ('umap', umap.UMAP(
                    n_components=2, 
                    random_state=10,
                    n_neighbors=nei,
                    min_dist=d
                ))
            ])
            run_pipeline(pipeline, d, nei, p)
            
            pca_step = pipeline.named_steps['pca']
            explained_var = pca_step.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)

            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_var)+1), explained_var, color='steelblue')

            x_values = range(1, len(explained_var) + 1)
            bars = plt.bar(x_values, explained_var, color='steelblue', alpha=0.7)

            plt.xlabel('Principal Component')
            plt.ylabel('Proportion of Variance Explained', color='steelblue')
            plt.title('PCA Explained Variance')
            plt.xticks(x_values)

            # Create a second y-axis for the cumulative variance
            ax2 = plt.twinx()
            ax2.plot(x_values, cumulative_var, color='darkred', marker='o')
            ax2.set_ylabel('Cumulative Variance Explained', color='darkred')
            ax2.tick_params(axis='y', labelcolor='darkred')

            plt.xlabel('Principal Component')
            plt.ylabel('Proportion of Variance Explained')
            plt.title('PCA Explained Variance')
            plt.tight_layout()
            plt.savefig(f"data/param_sweep/umap_param_sweep_{d}_{nei}_{p}.PCA.png", dpi=300)
            plt.close()  
            """
