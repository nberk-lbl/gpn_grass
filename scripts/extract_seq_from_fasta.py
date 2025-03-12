from Bio import SeqIO
import gzip
import sys
import os


#print("loading bed")
bed_intervals = {}

with open(sys.argv[1], 'r') as bed:
    for bed_line in bed:
        chrom, start, end = bed_line.split("\t")
        if not chrom in bed_intervals:
            bed_intervals[chrom] = []
        bed_intervals[chrom].append((int(start), int(end)))
        
#print("loading fasta")
with gzip.open(f'{sys.argv[2]}','rt') as genome_fa:
    for record in SeqIO.parse(genome_fa, "fasta"):
        if record.id in bed_intervals:
            intervals = bed_intervals[record.id]
            for interval in intervals:
                sequence = record.seq[interval[0] : interval[1]]
                print(f">{record.id},{interval[0]},{interval[1]}")
                print(str(sequence) + "\n")
                
#print(sequences)
