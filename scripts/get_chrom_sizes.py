from Bio import SeqIO
import sys
import gzip

with gzip.open(sys.argv[1], "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        print(f"{record.id}\t{str(len(record.seq))}")
