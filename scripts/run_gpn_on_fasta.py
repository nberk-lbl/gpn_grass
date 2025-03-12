
# run from inside gpn/gpn
import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import json

from Bio import SeqIO
import sys
import os
import gzip

os.environ['HF_HOME'] = '/pscratch/sd/n/nberk/gpn/gpn/tmp/'

print("loading model")
model_path = "songlab/gpn-brassicales"
model = AutoModelForMaskedLM.from_pretrained(model_path)

print("setting up tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.get_vocab()

print("loading model")
model = AutoModel.from_pretrained(model_path)
model.eval()

pfx = "/clusterfs/jgi/scratch/science/wcplant/nberkowitz/TAIR10_embeddings_"

sequences = ["1", "4", "5"]

print("loading fasta")
chunk_size = 100000  # adjust this to a suitable chunk size

with gzip.open(sys.argv[1], "rt") as genome_fa:
    for record in SeqIO.parse(genome_fa, "fasta"):
        if record.id in sequences:
            seq = str(record.seq)
            chunks = [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

            embeddings = []
            for chunk in chunks:
                input_ids = tokenizer(chunk, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]

                with torch.no_grad():
                    embedding = model(input_ids=input_ids).last_hidden_state
                embeddings.append(embedding[0].numpy())

            # concatenate the embeddings
            embedding = np.concatenate(embeddings, axis=0)

            # scale the embeddings
            embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding))
            embedding_df.to_csv(f'{pfx}{record.id}.tsv', sep='\t', index=False)
