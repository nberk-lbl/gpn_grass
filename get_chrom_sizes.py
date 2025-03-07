import sys
import os
from Bio import SeqIO
import gzip

path = "workflow/make_dataset/workflow/results/genome/"

for f in os.listdir(path):
    with gzip.open(f"{path}/{f}", 'rt') as fa, open(f"{f[0:-5]}chrom_sizes.txt", "w") as t:
        for record in SeqIO.parse(fa, "fasta"):
            print(f"{record.id}\t{len(record.seq)}", file=t)