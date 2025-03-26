import sys
from Bio import SeqIO
import random
import gzip


def filter_intergenic(gff_path, bed_intervals, n):

    # initialize data structures
    interval_labels = {}
    for (chrom, bed_start, bed_end) in bed_intervals:
        if not chrom in interval_labels:
            interval_labels[chrom] = {}
        interval_labels[chrom][(bed_start, bed_end)] = {}

    # count up the gff labeled bases
    i = 0
    with gzip.open(gff_path, 'rt') as gff_file:
        
        for line in gff_file:
            
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            chrom = cols[0]
            label = cols[2]
            gff_start = int(cols[3])
            gff_end = int(cols[4])

            for (bed_chrom, bed_start, bed_end) in bed_intervals:
                if chrom == bed_chrom:
                    if abs(bed_start - gff_start) < 100000:
                        overlap_start = max(gff_start, bed_start)
                        overlap_end = min(gff_end, bed_end)
                        if overlap_start <= overlap_end:
                            if not label in interval_labels:
                                interval_labels[chrom][(bed_start,bed_end)][label] = 0
                            interval_labels[chrom][(bed_start,bed_end)][label] += overlap_end - overlap_start
            i += 1
            if i % 100000 == 0:
                print(i)
            
    
    # get the intervals with the best score
    intervals_by_label_score = {}
    scores = []

    for chrom in interval_labels:
        for (bed_start, bed_end) in interval_labels[chrom]:
            score = 1
            for label in interval_labels[chrom][(bed_start, bed_end)]:
                score *= max(interval_labels[chrom][(bed_start, bed_end)][label], 1)
            scores.append(score)
            if not score in intervals_by_label_score:
                intervals_by_label_score[score] = []
            intervals_by_label_score[score].append((chrom, bed_start, bed_end))

    # take the top n
    scores.sort()
    scores.reverse()

    s = 0
    added = 0
    score = scores[s]
    intervals = intervals_by_label_score[score]

    filtered_intervals_index = {}
    filtered_intervals = []

    while added < n and s < len(scores):

        to_add = min(n - added, len(intervals))
        for a in range(to_add):
            chrom, start, end = intervals[a]
            if not chrom in filtered_intervals_index:
                filtered_intervals_index[chrom] = {}
            if not start in filtered_intervals_index[chrom]:
                filtered_intervals_index[chrom][start] = []
            filtered_intervals_index[chrom][start].append((chrom, start, end ))
            added += 1
        s += 1
        score = scores[s]
        intervals = intervals_by_label_score[score]

    chroms = list(filtered_intervals_index.keys())
    chroms.sort()
    for c in chroms:
        starts = list(filtered_intervals_index[c])
        starts.sort()
        for s in starts:
            for interval in filtered_intervals_index[c][s]:
                filtered_intervals.append(interval)

    print("-----")
    for i in filtered_intervals:
        print(i)
    return(filtered_intervals)

def make_gff(out_path, gff_path, bed_intervals):

    """Subset GFF file based on BED intervals (now 1-based)."""
    with gzip.open(gff_path, 'rt') as gff_file, gzip.open(out_path, 'wt') as out_gff:
        for line in gff_file:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9:
                continue
            chrom = cols[0]
            if chrom not in bed_intervals:
                continue  # Skip chromosomes not in BED
            try:
                gff_start = int(cols[3])
                gff_end = int(cols[4])
            except ValueError:
                continue  # Skip lines with non-integer start/end
            # Check overlaps with all BED intervals for this chromosome
            for (bed_start, bed_end) in bed_intervals[chrom]:
                overlap_start = max(gff_start, bed_start)
                overlap_end = min(gff_end, bed_end)
                if overlap_start <= overlap_end:
                    new_cols = cols.copy()
                    new_cols[0] = f"{chrom}:{bed_start-1}-{bed_end}"
                    new_cols[3] = str(overlap_start)
                    new_cols[4] = str(overlap_end)
                    print('\t'.join(new_cols), file=out_gff)


def make_fa(fa_path, intervals, indexed_sequences):
    
    with gzip.open(fa_path, 'wt') as out_fa:
        for i in intervals:
            chrom, start, end = i
            seq = indexed_sequences[chrom][start:end]
            print(f">{chrom}:{start}-{end}", file=out_fa)
            print(seq + "\n", file=out_fa)


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
    intervals_added = 0
    key_order = []

    lim = 0
    chroms = list(seq_by_chrom.keys())

    for c in chroms:
        intervals_by_chrom[c] = []
    weights = update_weights(seq_by_chrom, intervals_by_chrom, interval_length, chroms)

    while intervals_added < n_intervals and lim < 10000:
        chrom = random.choices(chroms, weights=weights, k=1)[0]
        s = len(seq_by_chrom[chrom])
        
        if (s > (interval_length * 2)):
            pos = random.randint(0, s - interval_length)
            end_pos = pos + interval_length    
            
            overlap = False
            for i in intervals_by_chrom[chrom]:
                start, end = i
                if pos > start and pos < end:
                    overlap = True

                if end_pos > start and end_pos < end:
                    overlap = True

            if not overlap:
                intervals_by_chrom[chrom].append((pos, end_pos))
                intervals_added += 1

            lim+=1
            weights = update_weights(seq_by_chrom, intervals_by_chrom, interval_length, chroms)

    return(sort_intervals(intervals_by_chrom))

def sort_intervals(intervals_by_chrom):
    intervals = []
    chroms = list(intervals_by_chrom.keys())
    chroms.sort()

    for chrom in chroms:
        start_index = {}
        ordered_pos = []
        for interval in intervals_by_chrom[chrom]:
            pos = interval[1]
            start_index[int(pos)] = list(interval)
            ordered_pos.append(int(pos))
        ordered_pos.sort()
        for s in ordered_pos:
            intervals.append([chrom] + start_index[s])

    return(intervals)


def index_fa_by_chrom(fa_gz):
    seq = {}
    with gzip.open(fa_gz, 'rt') as fasta:
        records = SeqIO.parse(fasta, "fasta")
        for rec in records:
            seq[rec.id] = rec.seq
    return(seq)

if __name__ == "__main__":
    random.seed(10)
    oversample = 20

    in_fa   = sys.argv[1]
    in_gff   = sys.argv[2]
    n_seq   = int(sys.argv[3])
    l_seq   = int(sys.argv[4])
    out_pfx = sys.argv[5]

    print("indexing")
    chrom_index = index_fa_by_chrom(sys.argv[1])
    print("indexing DONE")

    print("building intervals")
    intervals = make_sample_intervals(chrom_index, n_seq * oversample, l_seq)
    print("building intervals DONE")

    print("enriching intervals")
    intervals = filter_intergenic(in_gff, intervals, n_seq)
    print("enriching intervals DONE")
    
    intervals_for_gff = {}
    with open(f"{out_pfx}.sample.bed", 'w') as out_bed:
        for b in intervals:
            print(f"{b[0]}:{b[1]}-{b[2]}\t{b[1]}\t{b[2]}", file=out_bed)
            if not b[0] in intervals_for_gff:
                intervals_for_gff[b[0]] = []
            intervals_for_gff[b[0]].append((b[1]+1, b[2]))

    make_gff(f"{out_pfx}.sample.gff3.gz", in_gff, intervals_for_gff)
    make_fa(f"{out_pfx}.sample.fa.gz", intervals, chrom_index)
    


