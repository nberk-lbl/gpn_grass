import pandas as pd
import sys
import gzip

# Read the bed file into a dictionary
bed_intervals = {}
with open(sys.argv[1], 'r') as bed:
    for bed_line in bed:
        chrom, start, end = bed_line.split("\t")
        if not chrom in bed_intervals:
            bed_intervals[chrom] = []
        bed_intervals[chrom].append((int(start), int(end)))
print(bed_intervals)

# Process each chromosome separately
for chrom in bed_intervals:
    print(chrom)
    # Initialize the output file
    output_file = f'/clusterfs/jgi/scratch/science/wcplant/nberkowitz/{chrom}_labels.tsv'

    # Process each interval
    for start, end in bed_intervals[chrom]:
        print(start)
        
        # Create a dictionary to store the labels for each position
        labels_dict = {}

        # Iterate over the GFF file
        with gzip.open(sys.argv[2], 'rt') as gff:
            i = 0
            for line in gff:
                if i % 10000 == 1:
                    print(i)
                i+=1
                if line[0] != '#':
                    gff_chrom, a, category, gff_start, gff_end, b, strand, c, annot = line.rstrip().split("\t")
                    gff_start, gff_end = int(gff_start), int(gff_end)
                    if gff_chrom == chrom and gff_start <= end and gff_end >= start:
                        for pos in range(max(gff_start, start), min(gff_end + 1, end + 1)):
                            if pos in labels_dict:
                                labels_dict[pos].add(f"{category}({strand})")
                            else:
                                labels_dict[pos] = {f"{category}({strand})"}

        # Create a DataFrame to store the output
        output_df = pd.DataFrame(columns=['chromosome', 'pos', 'labels'])

        rows = []
        j = 0
        for pos in range(start, end + 1):
            if j % 10000 == 1:
                print(j)
            j+=1
            if pos in labels_dict:
                labels = ','.join(sorted(labels_dict[pos]))
            else:
                labels = ''
            rows.append({
                "chromosome": chrom,
                "pos": pos,
                "labels": labels
            })
            j += 1
        output_df = pd.DataFrame(rows)

        # Write the interval to the output file
        print("writing to file")
        if start == bed_intervals[chrom][0][0]:
            output_df.to_csv(output_file, sep='\t', index=False)
        else:
            output_df.to_csv(output_file, sep='\t', mode='a', header=False, index=False)
