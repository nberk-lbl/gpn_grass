import sys
from Bio import SeqIO
import random
import gzip


def check_intergenic():
    pass


def make_fa():
    pass

def update_weights(seq_by_chrom, intervals_by_chrom, interval_length, chrom_list):
    weights = []
    total = 0.
    total_interval = 0.

    for c in seq_by_chrom:
        total += len(seq_by_chrom[c])
    
    for c in intervals_by_chrom:
        total_interval += len(intervals_by_chrom[c]) * interval_length

    total_unused = total - total_interval

    for c in chrom_list:
        unused = len(seq_by_chrom[c]) - (len(intervals_by_chrom[c]) * interval_length)
        weights.append(unused/total_unused)

    return(weights)

def make_sample_intervals(seq_by_chrom, n_intervals, interval_length):
    intervals_by_chrom = {}
    intervals = []

    lim = 0
    chroms = list(seq_by_chrom.keys())

    for c in chroms:
        intervals_by_chrom[c] = []
    weights = update_weights(seq_by_chrom, intervals_by_chrom, interval_length, chroms)

    while len(intervals) < n_intervals and lim < 10000:
        chrom = random.choices(chroms, weights=weights, k=1)[0]
        s = len(seq_by_chrom[chrom])
        
        pos = random.randint(0, s - interval_length)
        end_pos = pos + interval_length    
        
        overlap = False
        for i in intervals_by_chrom[chrom]:
            print(i)
            start, end = i
            if pos > start and pos < end:
                overlap = True

            if end_pos > start and end_pos < end:
                overlap = True

        if not overlap:
            intervals.append((chrom, pos, end_pos))
            intervals_by_chrom[chrom].append((pos, end_pos))

        lim+=1
        weights = update_weights(seq_by_chrom, intervals_by_chrom, interval_length, chroms)
    return(intervals)

def index_fa_by_chrom(fa_gz):
    seq = {}
    with gzip.open(fa_gz, 'rt') as fasta:
        records = SeqIO.parse(fasta, "fasta")
        for rec in records:
            seq[rec.id] = rec.seq
    return(seq)

random.seed(10)
d = index_fa_by_chrom(sys.argv[1])
i = make_sample_intervals(d, 10, 1000)

for b in i:
    print(b)
