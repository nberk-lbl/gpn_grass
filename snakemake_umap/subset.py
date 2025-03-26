import gzip
from collections import defaultdict

def read_bed(bed_path):
    """Read BED file (0-based) and convert to 1-based intervals."""
    bed_intervals = defaultdict(list)
    with open(bed_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            chrom = parts[0]
            start_0based = int(parts[1])
            end_0based = int(parts[2])
            # Convert to 1-based [start, end] (inclusive)
            start = start_0based + 1
            end = end_0based
            bed_intervals[chrom].append((start, end))
    return bed_intervals

def subset_gff(gff_path, bed_intervals, debug=False):
    """Subset GFF while preserving original format, with overlap adjustments."""
    print(f"Debug: BED intervals loaded: {dict(bed_intervals)}", file=sys.stderr)  # Debug line
    
    with gzip.open(gff_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                print(line.strip())
                continue
            
            cols = line.strip().split('\t')
            if len(cols) < 9:
                if debug:
                    print(f"Skipping malformed line: {line.strip()}", file=sys.stderr)
                continue
            
            chrom = cols[0]
            if chrom not in bed_intervals:
                if debug:
                    print(f"Skipping chromosome not in BED: {chrom}", file=sys.stderr)
                continue
            
            try:
                gff_start = int(cols[3])
                gff_end = int(cols[4])
            except ValueError:
                if debug:
                    print(f"Skipping line with non-integer coordinates: {line.strip()}", file=sys.stderr)
                continue
            
            overlaps_found = False
            for (bed_start, bed_end) in bed_intervals[chrom]:
                # Calculate overlap (any intersection)
                overlap_start = max(gff_start, bed_start)
                overlap_end = min(gff_end, bed_end)
                
                if overlap_start <= overlap_end:
                    overlaps_found = True
                    new_cols = cols.copy()
                    new_cols[3] = str(overlap_start)
                    new_cols[4] = str(overlap_end)
                    print('\t'.join(new_cols))
            
            if debug and not overlaps_found:
                print(f"No overlap for GFF feature: {chrom} {gff_start}-{gff_end}", file=sys.stderr)

if __name__ == "__main__":
    import sys
    bed_path = "test.sample.bed"
    gff_path = "/pscratch/sd/n/nberk/results/annotation/Phallii_590_v3.2.gene_exons.gff3.gz"
    
    bed_intervals = read_bed(bed_path)
    subset_gff(gff_path, bed_intervals, debug=True)  # Enable debug mode
