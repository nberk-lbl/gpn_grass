import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


labels_file = "/pscratch/sd/n/nberk/grass/labels_subset.tsv"
embeddings_file = "/pscratch/sd/n/nberk/grass/embeddings.tsv"
set_size = 1000000

minimal_lables = []
with open(labels_file) as lbf, open("/pscratch/sd/n/nberk/grass/1m_labels.tsv", "w") as f:
    for n in range(set_size+1):
        print(lbf.readline().rstrip(), file=f)

minimal_embeddings = pd.read_csv(embeddings_file, delimiter='\t', nrows=set_size, dtype=np.float32)
minimal_embeddings.to_csv("/pscratch/sd/n/nberk/grass/1m_embeddings.tsv", sep="\t", index=False)
