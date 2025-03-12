import sys
from Bio import SeqIO
import os
import gzip

base_path = '/pscratch/sd/n/nberk'
hardmasked_fa_path = base_path + "/hardmasked_fa/"
out_bed_path = base_path + "/repeat_bed/"

files = os.listdir(hardmasked_fa_path)
for f in files:
    out_bed = out_bed_path + f[0:-16] + "repeat.bed"
    with gzip.open(hardmasked_fa_path + f, 'rt') as fa, open(out_bed, 'w') as bed:
        records = SeqIO.parse(fa, "fasta")
        intervals = {}
        start = 0
        end = 0
        chrom = ""
        inside_interval = False
        for rec in records:
            chrom = rec.id
            intervals[chrom] = []
            pos = 0
            for n in rec.seq:
                if n == "N":
                    if not inside_interval:
                        start = str(pos)
                        inside_interval = True
                else:
                    if inside_interval:
                        end = str(pos)
                        inside_interval = False
                        intervals[chrom].append((start, end))
                pos += 1

        for chrom in intervals:
            for interval in intervals[chrom]:
                print(f"{chrom}\t{interval[0]}\t{interval[1]}", file = bed)                    
                        
                        



