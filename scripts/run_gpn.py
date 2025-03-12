
# run from inside gpn/gpn
import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import json

from Bio import SeqIO
import gzip
import sys
import os


def rc(seq):

    comp = {
        'A' : 'T',
        'T' : 'A',
        'G' : 'C',
        'C' : 'G',
        'N' : 'N'
    }
    rc = ''

    rev = seq[::-1]
    for n in rev:
        c = n
        if n in comp:
            c = comp[n]
        rc += c

    return rc



os.environ['HF_HOME'] = '/pscratch/sd/n/nberk/gpn/gpn/tmp/'
run_cfg = json.loads(open(sys.argv[1], 'r').read())

print("loading model")
model_path = "songlab/gpn-brassicales"
model = AutoModelForMaskedLM.from_pretrained(model_path)

print("loading fasta")
sequences = {}
seq_order = []

with gzip.open(f'{run_cfg["fa_path"]}','rt') as genome_fa:
    for record in SeqIO.parse(genome_fa, "fasta"):
        if record.id in run_cfg['chrom_size']:
            sequences[record.id] = record.seq
            seq_order.append(record.id)

test_name = run_cfg['test_seq']
print(f"testing with {test_name}")

plt.figure(figsize=(10, 6))

print("setting up tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.get_vocab()

print("loading model")
model = AutoModel.from_pretrained(model_path)
model.eval()

window_size = run_cfg['window_size']
chunk_size = window_size * 1000
pfx = run_cfg['out_pfx']

final_averaged_embeddings = pd.DataFrame()
for rec in seq_order:
    
    seq = sequences[rec]
    
    for i in range(int(len(seq) / chunk_size) + 1):
        a = i * chunk_size
        b = min((i+1) * chunk_size, len(seq))
        
        print(f"computing {rec}: {a} - {b}")
        chunk = seq[a:b]
        input_ids = tokenizer(str(chunk), return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        print(input_ids.shape)

        with torch.no_grad():
            embedding = model(input_ids=input_ids).last_hidden_state
        print(embedding.shape)

        rc_chunk = rc(chunk)
        rc_input_ids = tokenizer(str(rc_chunk), return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        with torch.no_grad():
            rc_embedding = model(input_ids=rc_input_ids).last_hidden_state

        embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding[0].numpy()))
        rc_embeddings_df = pd.DataFrame(StandardScaler().fit_transform(rc_embedding[0].numpy()))

        # Reverse the order of rc_embeddings so they align with the forward embeddings
        rc_embeddings_df_reversed = rc_embeddings_df.iloc[::-1].reset_index(drop=True)

        # Average the embeddings from forward and reverse complement
        combined_embeddings = (embedding_df + rc_embeddings_df_reversed) / 2

        # embeddings across the window`
        averaged_embeddings = combined_embeddings.groupby(combined_embeddings.index // window_size).mean()

        final_averaged_embeddings = pd.concat([final_averaged_embeddings, averaged_embeddings], ignore_index=True)

final_averaged_embeddings.to_csv(f'{run_cfg["output_dir"]}/{pfx}.avg_embeddings_N.tsv', sep='\t', index=False)
